#!/usr/bin/env python3
import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from torchvision import transforms

from gnn.patientgraphmodel import PatientGraphModel
from data_functions.BRSETGraphDataset import BRSETGraphDataset
from utils.data_functions import create_dataloader, calculate_positive_weights
from utils.feature_engineering import (
    preprocess_comorbidities,
    infer_demographic_columns,
    fit_demographic_specs
)
from utils.logger import start_wandb_run
from utils.metrics import find_optimal_thresholds
from utils.train_functions import graph_epoch
from utils.util_functions import select_device, set_seed, make_grad_scaler, write_json

try:
    import wandb
except ImportError:
    wandb = None

# =====================================================================
# 1. 7 TARGET LABELS CONFIGURATION
# =====================================================================

ALL_LABELS = [
    "increased_cup_disc", "drusens", "diabetic_retinopathy", "macular_edema",
    "scar", "amd", "myopic_fundus"
]

# =====================================================================
# 2. CLI ARGUMENT PARSER
# =====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the Multilabel Image-Only Model (7 Labels) with W&B logging."
    )
    # Required/Custom Model Name
    p.add_argument(
        "--model-name", 
        type=str, 
        required=True, 
        help="Name of the model run (used for checkpoint directory and W&B run name)."
    )
    
    # Dataset and Output paths
    p.add_argument("--root", type=Path, default=Path("data/BRSET"), help="Dataset root directory.")
    p.add_argument("--prepared-dir", type=Path, default=None, help="Directory containing train.csv/val.csv.")
    p.add_argument("--train-manifest", type=str, default="image_train_12.csv")
    p.add_argument("--validation-manifest", type=str, default="image_validation_12.csv")
    p.add_argument("--output-dir", type=Path, default=None, help="Root directory for saving checkpoints.")
    
    # Vision & Graph Setup
    p.add_argument("--backbone", type=str, choices=["retfound", "resnet"], default="resnet")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--patch-size", type=int, default=None, help="Defaults to 32 for resnet, 16 for retfound")
    p.add_argument("--connectivity", default="fc", help="Patch graph connectivity: 'fc' (dense SDPA with Gaussian bias, recommended), 4, or 8 (defaults to fc)")
    
    # Model Architecture (Matched to tune_all defaults)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.25)
    
    # Optimization & Training
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--micro-batch-size", type=int, default=None, help="Physical mini-batch size (defaults to batch_size).")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--compile", action="store_true", help="Compile model with torch.compile (disabled by default due to PyG HeteroConv recompilations).")

    # Weights & Biases (Defaulted to 'model update tests')
    p.add_argument("--wandb-project", type=str, default="model update tests", help="W&B project name.")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-group", type=str, default="Multilabel-Image-Only")
    p.add_argument("--wandb-tags", nargs="*", default=["multilabel_img_only", "7_labels", "dual_hub"])
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])

    return p.parse_args()


# =====================================================================
# 3. MAIN TRAINING ROUTINE
# =====================================================================

def main() -> None:
    args = parse_args()
    
    if args.patch_size is None:
        args.patch_size = 16 if args.backbone == "retfound" else 32

    root = args.root.expanduser().resolve()
    prepared_dir = args.prepared_dir.expanduser().resolve() if args.prepared_dir else (root / "prepared")
    base_output_dir = args.output_dir.expanduser().resolve() if args.output_dir else (root / "runs" / "model_update_tests")
    
    run_dir = base_output_dir / args.model_name
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed, args.deterministic)
    device = select_device(args.device)
    amp_enabled = not args.no_amp and device.type == "cuda"
    
    # Adaptive micro-batching and gradient accumulation to prevent CUDA OOM
    if args.micro_batch_size is not None:
        micro_bs = args.micro_batch_size
    elif args.hidden_dim >= 512:
        micro_bs = max(1, args.batch_size // 2)
    else:
        micro_bs = args.batch_size
        
    accum_steps = max(1, args.batch_size // micro_bs)

    print("===================================================================")
    print(f" TRAINING: {args.model_name.upper()} (Multilabel Image-Only)")
    print("===================================================================")
    print(f"Target Labels (7): {ALL_LABELS}")
    print(f"Architecture: hidden_dim={args.hidden_dim}, num_layers={args.num_layers}, heads={args.heads}, dropout={args.dropout}")
    print(f"Prediction Level: dual_hub (Direct Label Node Classification)")
    print(f"Batch Size: effective={args.batch_size}, micro_batch={micro_bs}, accum_steps={accum_steps}")
    print(f"Device: {device} | AMP: {amp_enabled} | Backbone: {args.backbone}")
    print(f"W&B Project: '{args.wandb_project}' | Run Name: '{args.model_name}'")
    print(f"Run Output Directory: {run_dir}\n")

    # Load DataFrames
    train_frame = pd.read_csv(prepared_dir / args.train_manifest)
    val_frame = pd.read_csv(prepared_dir / args.validation_manifest)
    
    # Comorbidity extraction
    train_frame, top_comorb = preprocess_comorbidities(train_frame, top_k=20)
    val_frame, _ = preprocess_comorbidities(val_frame, valid_comorb=top_comorb)

    # Demographic specs
    demo_columns = infer_demographic_columns(train_frame, None, ALL_LABELS)
    demo_specs = fit_demographic_specs(train_frame, demo_columns)

    # Vision Transforms
    if args.backbone == "resnet":
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Dataset Setup
    dataset_kwargs = {
        "image_dir": root / "fundus_photos_512",
        "target_cols": ALL_LABELS,
        "demo_cols": demo_columns,
        "demo_specs": demo_specs,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "connectivity": args.connectivity,
        "prediction_level": "dual_hub",
        "train_frame": train_frame
    }

    train_dataset = BRSETGraphDataset(
        data=train_frame, validate_paths=True, transform=train_transform, **dataset_kwargs
    )
    val_dataset = BRSETGraphDataset(
        data=val_frame, validate_paths=False, transform=val_transform, **dataset_kwargs
    )

    train_loader = create_dataloader(
        dataset=train_dataset, batch_size=micro_bs, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        seed=args.seed, is_graph=True
    )
    val_loader = create_dataloader(
        dataset=val_dataset, batch_size=micro_bs, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        seed=args.seed, is_graph=True
    )

    # Positive class re-weighting
    train_targets = np.vstack([s["label_vector"] for s in train_dataset.samples])
    pos_weight = calculate_positive_weights(train_targets).to(device)

    pos_counts = np.maximum(train_targets.sum(axis=0), 1.0)
    neg_counts = np.maximum(train_targets.shape[0] - pos_counts, 1.0)
    adaptive_biases = np.log(pos_counts / neg_counts).astype(np.float32)

    # Instantiate Model
    model = PatientGraphModel(
        num_labels=len(ALL_LABELS),
        demographic_specs=demo_specs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        heads=args.heads,
        prediction_level="dual_hub",
        initial_biases=adaptive_biases,
        use_context=False,          # Image-Only model baseline
        use_image_features=True,    # Image features enabled
        backbone=args.backbone,
        patch_connectivity=args.connectivity,
        image_size=args.image_size,
        patch_size=args.patch_size,
    ).to(device)

    if getattr(args, "compile", False) and os.name != 'nt':
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Skipping torch.compile: {e}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = make_grad_scaler(amp_enabled)

    warmup_epochs = max(1, int(0.1 * args.epochs))
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)

    # Save Configuration
    run_config = {
        "model_name": args.model_name,
        "experiment": "multilabel_img_only",
        "labels": ALL_LABELS,
        "no_context": True,
        "no_image": False,
        "prediction_level": "dual_hub",
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "heads": args.heads,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "micro_batch_size": micro_bs,
        "accum_steps": accum_steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "backbone": args.backbone,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "connectivity": args.connectivity,
        "train_manifest": args.train_manifest,
        "validation_manifest": args.validation_manifest
    }
    write_json(run_dir / "run_config.json", run_config)

    # Initialize Weights & Biases
    wandb_run = start_wandb_run(
        project=args.wandb_project,
        run_name=args.model_name,
        config=run_config,
        output_dir=run_dir,
        entity=args.wandb_entity,
        group=args.wandb_group,
        tags=args.wandb_tags,
        mode=args.wandb_mode
    )

    history = []
    best_metrics = {
        "macro_auprc": -1.0,
        "macro_auroc": -1.0,
        "macro_f1_calibrated": -1.0,
        "min_val_loss": float("inf")
    }

    checkpoint_path = run_dir / "final_checkpoint.pt"

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        train_res = graph_epoch(
            model=model, loader=train_loader, criterion=criterion, device=device,
            amp_enabled=amp_enabled, label_columns=ALL_LABELS, optimizer=optimizer,
            scaler=scaler, desc=f"[{args.model_name}] E{epoch:02d} Train", threshold=args.threshold,
            accum_steps=accum_steps
        )
        scheduler.step()

        val_res = graph_epoch(
            model=model, loader=val_loader, criterion=criterion, device=device,
            amp_enabled=amp_enabled, label_columns=ALL_LABELS, optimizer=None,
            scaler=None, desc=f"[{args.model_name}] E{epoch:02d} Val", threshold=args.threshold
        )

        best_thresholds, epoch_threshold_df = find_optimal_thresholds(
            targets=val_res["targets"],
            probabilities=val_res["probabilities"],
            label_names=ALL_LABELS,
            step=0.05
        )
        epoch_calibrated_f1 = float(epoch_threshold_df["validation_f1"].mean())
        epoch_balanced_acc = float(epoch_threshold_df["validation_balanced_accuracy"].mean())

        calibrated_predictions = (val_res["probabilities"] >= best_thresholds.reshape(1, -1)).astype(int)
        epoch_calibrated_micro_f1 = float(f1_score(val_res["targets"], calibrated_predictions, average="micro", zero_division=0))

        history.append({
            "epoch": epoch,
            "train_loss": train_res["loss"],
            "val_loss": val_res["loss"],
            "val_macro_f1_static": val_res["summary"]["macro_f1"],
            "val_micro_f1_static": val_res["summary"]["micro_f1"],
            "val_macro_f1_calibrated": epoch_calibrated_f1,
            "val_micro_f1_calibrated": epoch_calibrated_micro_f1,
            "val_macro_auroc": val_res["summary"]["macro_auroc"],
            "val_macro_auprc": val_res["summary"]["macro_auprc"]
        })
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        current_macro_auprc = val_res["summary"]["macro_auprc"]

        raw_model = getattr(model, "_orig_mod", model)
        cnn_has_trainable = hasattr(raw_model, "cnn") and any(p.requires_grad for p in raw_model.cnn.parameters())
        if args.backbone == "retfound" and not cnn_has_trainable:
            state_dict_to_save = {k: v for k, v in raw_model.state_dict().items() if not k.startswith("cnn.")}
        else:
            state_dict_to_save = raw_model.state_dict()

        checkpoint_data = {
            "model_state_dict": state_dict_to_save,
            "hyperparams": run_config,
            "epoch": epoch
        }
        torch.save(checkpoint_data, checkpoint_path)

        if current_macro_auprc > best_metrics["macro_auprc"]:
            best_metrics["macro_auprc"] = current_macro_auprc
            torch.save(checkpoint_data, run_dir / "checkpoint_best_macro_auprc.pt")
            if wandb_run is not None:
                wandb_run.summary["best_macro_auprc"] = current_macro_auprc
                wandb_run.summary["best_macro_auprc_epoch"] = epoch

        if epoch_calibrated_f1 > best_metrics["macro_f1_calibrated"]:
            best_metrics["macro_f1_calibrated"] = epoch_calibrated_f1
            torch.save(checkpoint_data, run_dir / "checkpoint_best_macro_f1.pt")

        if wandb_run is not None:
            y_true = val_res["targets"]
            y_prob = val_res["probabilities"]

            log_dict = {
                "epoch": epoch,
                "lr": scheduler.get_last_lr()[0],
                "train/loss": train_res["loss"],
                "val/loss": val_res["loss"],
                "val/macro_f1_uncalibrated": val_res["summary"]["macro_f1"],
                "val/micro_f1_uncalibrated": val_res["summary"]["micro_f1"],
                "val/macro_f1_calibrated": epoch_calibrated_f1,
                "val/micro_f1_calibrated": epoch_calibrated_micro_f1,
                "val/macro_auroc": val_res["summary"]["macro_auroc"],
                "val/macro_auprc": val_res["summary"]["macro_auprc"],
                "val/macro_accuracy_calibrated": float(epoch_threshold_df["validation_accuracy"].mean()),
                "val/macro_balanced_acc_calibrated": epoch_balanced_acc,
            }

            uncalibrated_preds = (y_prob >= 0.5).astype(int)
            for idx, label in enumerate(ALL_LABELS):
                prefix = f"{idx+1:02d}_"
                try:
                    auroc = float(roc_auc_score(y_true[:, idx], y_prob[:, idx]))
                except ValueError:
                    auroc = 0.0
                try:
                    auprc = float(average_precision_score(y_true[:, idx], y_prob[:, idx]))
                except ValueError:
                    auprc = 0.0

                calib_f1 = float(f1_score(y_true[:, idx], calibrated_predictions[:, idx], average="binary", zero_division=0))
                uncalib_f1 = float(f1_score(y_true[:, idx], uncalibrated_preds[:, idx], average="binary", zero_division=0))

                log_dict[f"val_auroc_per_class/{prefix}{label}"] = auroc
                log_dict[f"val_auprc_per_class/{prefix}{label}"] = auprc
                log_dict[f"val_calibrated_f1_per_class/{prefix}{label}"] = calib_f1
                log_dict[f"val_uncalibrated_f1_per_class/{prefix}{label}"] = uncalib_f1
                log_dict[f"val_optimal_threshold/{prefix}{label}"] = float(epoch_threshold_df.iloc[idx]["optimal_threshold"])

            wandb.log(log_dict, step=epoch)

    # 4. Final Best AUPRC Checkpoint Re-evaluation
    best_auprc_path = run_dir / "checkpoint_best_macro_auprc.pt"
    if best_auprc_path.exists():
        ckpt = torch.load(best_auprc_path, map_location=device)
        raw_model = getattr(model, "_orig_mod", model)
        raw_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"\n--> Loaded best AUPRC checkpoint from epoch {ckpt.get('epoch', 'N/A')}. Re-running validation evaluation...")

    final_val_res = graph_epoch(
        model=model, loader=val_loader, criterion=criterion, device=device,
        amp_enabled=amp_enabled, label_columns=ALL_LABELS, optimizer=None,
        scaler=None, desc=f"[{args.model_name}] Final Best Val Eval", threshold=args.threshold
    )

    best_thresholds, final_threshold_df = find_optimal_thresholds(
        targets=final_val_res["targets"],
        probabilities=final_val_res["probabilities"],
        label_names=ALL_LABELS,
        step=0.05
    )

    final_calibrated_f1 = float(final_threshold_df["validation_f1"].mean())
    final_calibrated_acc = float(final_threshold_df["validation_accuracy"].mean())
    final_balanced_acc = float(final_threshold_df["validation_balanced_accuracy"].mean())
    
    final_calibrated_preds = (final_val_res["probabilities"] >= best_thresholds.reshape(1, -1)).astype(int)
    final_calibrated_micro_f1 = float(f1_score(final_val_res["targets"], final_calibrated_preds, average="micro", zero_division=0))

    final_eval_summary = {
        "best_epoch": ckpt.get("epoch", None) if best_auprc_path.exists() else args.epochs,
        "final_val_loss": float(final_val_res["loss"]),
        "final_val_macro_auprc": float(final_val_res["summary"]["macro_auprc"]),
        "final_val_macro_auroc": float(final_val_res["summary"]["macro_auroc"]),
        "final_val_macro_f1_calibrated": final_calibrated_f1,
        "final_val_micro_f1_calibrated": final_calibrated_micro_f1,
        "final_val_accuracy_calibrated": final_calibrated_acc,
        "final_val_balanced_acc_calibrated": final_balanced_acc,
        "per_class": {}
    }

    y_true = final_val_res["targets"]
    y_prob = final_val_res["probabilities"]
    for idx, label in enumerate(ALL_LABELS):
        prefix = f"{idx+1:02d}_"
        try:
            auroc = float(roc_auc_score(y_true[:, idx], y_prob[:, idx]))
        except ValueError:
            auroc = 0.0
        try:
            auprc = float(average_precision_score(y_true[:, idx], y_prob[:, idx]))
        except ValueError:
            auprc = 0.0

        calib_f1 = float(f1_score(y_true[:, idx], final_calibrated_preds[:, idx], average="binary", zero_division=0))
        opt_thresh = float(final_threshold_df.iloc[idx]["optimal_threshold"])

        final_eval_summary["per_class"][label] = {
            "auprc": auprc,
            "auroc": auroc,
            "calibrated_f1": calib_f1,
            "optimal_threshold": opt_thresh
        }

    print("\n" + "=" * 65)
    print(f" MODEL RUN COMPLETED: {args.model_name.upper()}")
    print(f" Best Checkpoint (by AUPRC) from Epoch {final_eval_summary['best_epoch']}:")
    print(f"   • Validation Macro AUPRC:         {final_eval_summary['final_val_macro_auprc']:.4f}")
    print(f"   • Validation Macro AUROC:         {final_eval_summary['final_val_macro_auroc']:.4f}")
    print(f"   • Validation Calibrated Macro F1: {final_eval_summary['final_val_macro_f1_calibrated']:.4f}")
    print(f"   • Validation Calibrated Micro F1: {final_eval_summary['final_val_micro_f1_calibrated']:.4f}")
    print(f"   • Validation Loss:                 {final_eval_summary['final_val_loss']:.4f}")
    print("=" * 65 + "\n")

    write_json(run_dir / "best_val_eval_results.json", final_eval_summary)

    if wandb_run is not None:
        wandb_run.summary["best_val/epoch"] = final_eval_summary["best_epoch"]
        wandb_run.summary["best_val/loss"] = final_eval_summary["final_val_loss"]
        wandb_run.summary["best_val/macro_auprc"] = final_eval_summary["final_val_macro_auprc"]
        wandb_run.summary["best_val/macro_auroc"] = final_eval_summary["final_val_macro_auroc"]
        wandb_run.summary["best_val/macro_f1_calibrated"] = final_eval_summary["final_val_macro_f1_calibrated"]
        wandb_run.summary["best_val/micro_f1_calibrated"] = final_eval_summary["final_val_micro_f1_calibrated"]

        for idx, label in enumerate(ALL_LABELS):
            prefix = f"{idx+1:02d}_"
            c_stats = final_eval_summary["per_class"][label]
            wandb_run.summary[f"best_val_per_class/auprc_{prefix}{label}"] = c_stats["auprc"]
            wandb_run.summary[f"best_val_per_class/f1_calibrated_{prefix}{label}"] = c_stats["calibrated_f1"]

        wandb_run.finish()

    del model, optimizer, train_loader, val_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\nTraining for '{args.model_name}' complete! Best Macro AUPRC: {best_metrics['macro_auprc']:.4f}")


if __name__ == "__main__":
    main()
