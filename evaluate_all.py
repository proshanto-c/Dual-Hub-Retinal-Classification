import argparse
import os
import glob
from pathlib import Path
import json
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score

from data_functions.BRSETGraphDataset import BRSETGraphDataset
from gnn.patientgraphmodel import PatientGraphModel, DemographicSpec
from utils.data_functions import create_dataloader
from utils.train_functions import graph_epoch
from utils.util_functions import select_device, set_seed
from utils.feature_engineering import preprocess_comorbidities, infer_demographic_columns, fit_demographic_specs

# Hardcoded base labels and demographics for the schema
ALL_LABELS = [
    "increased_cup_disc", "drusens", "diabetic_retinopathy", "macular_edema",
    "scar", "amd", "myopic_fundus"
]

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a directory of trained models.")
    p.add_argument("--root", type=Path, default="data/BRSET", help="Dataset root directory.")
    p.add_argument("--prepared-dir", type=Path, default=None, help="Directory containing test.csv")
    p.add_argument("--test-manifest", type=str, default="image_test_12.csv")
    p.add_argument("--runs-dir", type=Path, required=True, help="Path to the directory containing model subfolders (e.g., tuning_seed42).")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--checkpoint-name", type=str, default=None, help="Force a specific checkpoint to load (e.g. checkpoint.pt)")
    return p.parse_args()


def load_demographic_specs(hyperparams):
    # Depending on how it was saved, hyperparams might contain demo_specs as a dict.
    # Reconstruct DemographicSpec objects.
    specs = []
    if "demo_specs" in hyperparams:
        for ds in hyperparams["demo_specs"]:
            if isinstance(ds, dict):
                specs.append(DemographicSpec(**ds))
            else:
                specs.append(ds) # Already an object
    return specs


def evaluate_model(config_dir, args, device, test_transform, test_frame, checkpoint_path, fallback_demo_specs, train_frame: pd.DataFrame):
    if not checkpoint_path.exists():
        print(f"[{config_dir.name}] Skipping... No checkpoint found at {checkpoint_path}.")
        return None
                
    print(f"\nEvaluating: {config_dir.name} (using {checkpoint_path.name})")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hyperparams = ckpt.get("hyperparams", {})
    experiment_name = hyperparams.get("experiment", config_dir.name)
    
    label_columns = hyperparams.get("labels", ALL_LABELS)
    demo_columns = hyperparams.get("demo_cols", [])
    demo_specs = load_demographic_specs(hyperparams)
    if not demo_specs:
        print("  -> Warning: demo_specs missing from checkpoint. Using dynamically recalculated fallback!")
        demo_specs = fallback_demo_specs
        
    if not demo_columns:
        demo_columns = [ds.name for ds in demo_specs]
    
    # Ensure all demo_columns exist in test_frame
    test_frame = test_frame.copy()
    for col in demo_columns:
        if col not in test_frame.columns:
            test_frame[col] = 0.0
    
    # Defaults in case the checkpoint doesn't have them
    h_dim = hyperparams.get("h_dim", 256)
    n_layers = hyperparams.get("n_layers", 3)
    dropout = hyperparams.get("dropout", 0.0)
    backbone = hyperparams.get("backbone", "resnet")
    patch_size = hyperparams.get("patch_size", 32)
    connectivity = hyperparams.get("connectivity", "fc")
    no_context = hyperparams.get("no_context", False)
    no_image = hyperparams.get("no_image", False)
    
    dataset_kwargs = {
        "image_dir": args.root.expanduser().resolve() / "fundus_photos_512",
        "target_cols": label_columns,
        "demo_cols": demo_columns,
        "demo_specs": demo_specs,
        "image_size": args.image_size,
        "patch_size": patch_size,
        "connectivity": connectivity,
        "prediction_level": "dual_hub",
        "train_frame": train_frame
    }
    
    norm_step = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])] if backbone == "resnet" else []
    eval_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        *norm_step
    ])
    
    test_dataset = BRSETGraphDataset(
        data=test_frame,
        validate_paths=False,
        transform=eval_transform,
        **dataset_kwargs
    )
    
    test_loader = create_dataloader(
        dataset=test_dataset, batch_size=32, shuffle=False,
        num_workers=4, pin_memory=(device.type == "cuda"),
        seed=42, is_graph=True
    )
    
    # Initialize the model
    model = PatientGraphModel(
        num_labels=len(label_columns),
        demographic_specs=demo_specs,
        hidden_dim=h_dim,
        num_layers=n_layers,
        dropout=dropout,
        prediction_level="dual_hub",
        use_context=not no_context,
        use_image_features=not no_image,
        backbone=backbone,
        patch_connectivity=connectivity,
        image_size=args.image_size,
        patch_size=patch_size,
    ).to(device)
    
    # Strip '_orig_mod.' prefix if checkpoint was saved from a compiled model
    raw_state_dict = ckpt.get("model_state_dict", {})
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in raw_state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    # Run Evaluation
    res = graph_epoch(
        model=model, loader=test_loader, criterion=torch.nn.BCEWithLogitsLoss(),
        device=device, amp_enabled=False, label_columns=label_columns,
        optimizer=None, scaler=None, desc="Testing"
    )
    
    # Calculate Metrics per Label
    y_true = res["targets"]
    y_prob = res["probabilities"]
    
    y_pred_uncalibrated = (y_prob >= 0.5).astype(int)
    
    metrics = {}
    for idx, label in enumerate(label_columns):
        true_lbl = y_true[:, idx]
        prob_lbl = y_prob[:, idx]
        pred_lbl = y_pred_uncalibrated[:, idx]
        
        try:
            auroc = roc_auc_score(true_lbl, prob_lbl)
            auprc = average_precision_score(true_lbl, prob_lbl)
        except ValueError:
            auroc = 0.0
            auprc = 0.0
            
        f1 = f1_score(true_lbl, pred_lbl, zero_division=0)
        precision = precision_score(true_lbl, pred_lbl, zero_division=0)
        recall = recall_score(true_lbl, pred_lbl, zero_division=0)
        acc = accuracy_score(true_lbl, pred_lbl)
        bal_acc = balanced_accuracy_score(true_lbl, pred_lbl)
        
        metrics[label] = {
            "AUPRC": auprc,
            "AUROC": auroc,
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
            "Acc": acc,
            "Bal_Acc": bal_acc
        }
    
    return metrics, experiment_name


def main():
    args = parse_args()
    device = select_device(args.device)
    
    root = args.root.expanduser().resolve()
    prepared_dir = args.prepared_dir.expanduser().resolve() if args.prepared_dir else (root / "prepared")
    
    test_manifest_path = Path(args.test_manifest) if Path(args.test_manifest).is_absolute() else prepared_dir / args.test_manifest
    test_frame = pd.read_csv(test_manifest_path)
    
    # Calculate fallback demo_specs for old checkpoints & preprocess comorbidities on test_frame
    train_manifest_path = prepared_dir / "image_train_12.csv"
    if not train_manifest_path.exists():
        raise FileNotFoundError(f"Training manifest not found at {train_manifest_path}. Evaluation strictly requires train_frame to compute training statistics and Pearson correlations.")
    train_frame = pd.read_csv(train_manifest_path)
    train_frame, top_comorb = preprocess_comorbidities(train_frame, top_k=20)
    test_frame, _ = preprocess_comorbidities(test_frame, valid_comorb=top_comorb)
    fallback_demo_cols = infer_demographic_columns(train_frame, None, ALL_LABELS)
    fallback_demo_specs = fit_demographic_specs(train_frame, fallback_demo_cols)
    
    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])
    
    runs_dir = args.runs_dir.expanduser().resolve()
    model_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    
    all_results = {}
    
    for config_dir in sorted(model_dirs):
        # Find all 'best' checkpoints to evaluate recursively (since they are nested in trial_* folders)
        checkpoints = list(config_dir.rglob("checkpoint_best_*.pt"))
        if not checkpoints:
            print(f"[{config_dir.name}] No optimal checkpoints found, skipping.")
            continue
            
        for ckpt_path in checkpoints:
            try:
                res = evaluate_model(config_dir, args, device, test_transform, test_frame, ckpt_path, fallback_demo_specs, train_frame=train_frame)
                if res is not None:
                    metrics, exp_name = res
                    
                    if exp_name not in all_results:
                        all_results[exp_name] = {}
                        
                    ckpt_key = ckpt_path.stem.replace("checkpoint_best_", "")
                    all_results[exp_name][ckpt_key] = metrics
                    
                    # Print local results
                    print(f" --- {exp_name.upper()} : {ckpt_key.upper()} CHECKPOINT ---")
                    for label, vals in metrics.items():
                        print(f"  -> {label}: AUPRC={vals['AUPRC']:.4f} | F1={vals['F1']:.4f} | Prec={vals['Precision']:.4f} | Rec={vals['Recall']:.4f} | BalAcc={vals['Bal_Acc']:.4f}")
            except Exception as e:
                print(f"[{config_dir.name}] Failed to evaluate {ckpt_path.name}: {e}")
            
    # Save the consolidated JSON to the root runs_dir
    out_file = runs_dir / "test_evaluation_results.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\nConsolidated results saved to: {out_file}")

if __name__ == "__main__":
    main()
