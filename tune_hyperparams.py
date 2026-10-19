import argparse
import gc
import json
import os
import random
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Any, Optional, Sequence

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
# 1. METADATA & HYPERPARAMETER SEARCH SPACE
# =====================================================================

ALL_MULTILABEL_LABELS = [
    "increased_cup_disc", "drusens", "diabetic_retinopathy", "macular_edema",
    "scar", "amd", "myopic_fundus"
]
SINGLELABEL_DR_LABELS = ["diabetic_retinopathy"]
SINGLELABEL_MF_LABELS = ["myopic_fundus"]

LABEL_ALIASES = {
    "dr": "diabetic_retinopathy",
    "diabetic_retinopathy": "diabetic_retinopathy",
    "mf": "myopic_fundus",
    "myopic_fundus": "myopic_fundus",
    "myopia": "myopic_fundus",
    "icd": "increased_cup_disc",
    "glaucoma": "increased_cup_disc",
    "increased_cup_disc": "increased_cup_disc",
    "drusen": "drusens",
    "drusens": "drusens",
    "me": "macular_edema",
    "macular_edema": "macular_edema",
    "scar": "scar",
    "amd": "amd",
}

LABEL_SHORT_NAMES = {
    "diabetic_retinopathy": "DR",
    "myopic_fundus": "MF",
    "increased_cup_disc": "ICD",
    "drusens": "Drusens",
    "macular_edema": "ME",
    "scar": "Scar",
    "amd": "AMD",
}

GRID_SEARCH_SPACE = {
    "hidden_dim": [128, 256, 512],
    "dropout": [0.15, 0.25, 0.35],
    "lr_strategy": ["uniform", "split"],
}

# =====================================================================
# 2. CLI ARGUMENT PARSER
# =====================================================================

def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hyperparameter tuning for Image-Only PatientGraphModel (Singlelabel DR, MF, or Multilabel)."
    )
    p.add_argument("--root", type=Path, required=True, help="Dataset root directory.")
    p.add_argument("--prepared-dir", type=Path, default=None, help="Directory containing train.csv/val.csv.")
    
    p.add_argument("--train-manifest", type=str, default="image_train_12.csv")
    p.add_argument("--validation-manifest", type=str, default="image_validation_12.csv")
    
    p.add_argument("--output-dir", type=Path, default=None, help="Output directory for runs and checkpoints.")
    p.add_argument("--backbone", type=str, choices=["retfound", "resnet"], default="resnet")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--patch-size", type=int, default=None, help="Defaults to 16 for retfound, 32 for resnet")
    p.add_argument("--connectivity", default="fc", help="Patch graph connectivity: 'fc' (dense SDPA with Gaussian bias, recommended), 4, or 8 (defaults to fc)")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-4, help="Base GNN learning rate (default: 1e-4).")
    p.add_argument("--cnn-learning-rate", type=float, default=1e-5, help="CNN backbone learning rate when split (default: 1e-5).")

    # Target task / label toggles
    p.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target to tune: 'dr' (middle prevalence, default), 'mf' (low prevalence), 'multilabel', or any specific condition name.",
    )
    p.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Target mode: 'singlelabel_dr', 'singlelabel_mf', 'multilabel'.",
    )
    p.add_argument(
        "--singlelabel-dr",
        action="store_true",
        help="Convenience toggle: train singlelabel Diabetic Retinopathy models (middle prevalence, default).",
    )
    p.add_argument(
        "--singlelabel-mf", "--singlelabel-myopic-fundus",
        action="store_true",
        dest="singlelabel_mf",
        help="Convenience toggle: train singlelabel Myopic Fundus models (low prevalence).",
    )
    p.add_argument(
        "--multilabel",
        action="store_true",
        help="Convenience toggle: train multilabel models across all 7 conditions.",
    )
    p.add_argument(
        "--singlelabel",
        type=str,
        default=None,
        metavar="LABEL_NAME",
        help="Train a single-label model for any specified condition (e.g. dr, mf, amd, scar).",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Explicit list of target label(s) for tuning (overrides other toggles).",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--compile", action="store_true", help="Compile model with torch.compile (disabled by default due to PyG HeteroConv recompilations).")

    # Search strategy controls
    p.add_argument("--grid-search", action="store_true", help="Run full grid search over the search space (18 trials).")
    p.add_argument("--num-trials", type=int, default=5, help="Number of random hyperparameter trials (ignored if --grid-search is set).")
    
    # Specific hyperparameter overrides (for single trial or W&B sweeps)
    p.add_argument("--hidden-dim", type=int, choices=[128, 256, 512], default=None, help="Hidden dim (128, 256, 512).")
    p.add_argument("--dropout", type=float, choices=[0.15, 0.25, 0.35], default=None, help="Dropout probability (.15, .25, .35).")
    p.add_argument("--lr-strategy", type=str, choices=["uniform", "split"], default=None, help="'uniform' (1e-4 all) or 'split' (1e-4 gnn, 1e-5 cnn).")
    p.add_argument("--num-layers", type=int, default=3, help="Number of GNN layers (fixed to 3 for this sweep).")
    p.add_argument("--heads", type=int, default=2, help="Number of attention heads (default: 2).")

    # Splitting Arguments for SLURM Job Arrays
    p.add_argument("--split-total", type=int, default=1, help="Total number of parallel jobs.")
    p.add_argument("--split-index", type=int, default=0, help="0-indexed chunk to run.")

    # W&B Arguments
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name-prefix", type=str, default=None)
    p.add_argument("--wandb-group", type=str, default=None)
    p.add_argument("--wandb-tags", nargs="*", default=None)
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])

    return p.parse_args(args)


# =====================================================================
# 3. HELPER FUNCTIONS
# =====================================================================

def generate_trial_configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Generates the list of hyperparameter dictionary combinations to run."""
    # Check if single explicit hyperparameter set was provided via CLI
    if all(v is not None for v in [args.hidden_dim, args.dropout, args.lr_strategy]):
        return [{
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr_strategy": args.lr_strategy,
            "num_layers": args.num_layers,
            "heads": args.heads,
        }]
    
    # Generate all 18 combinations in the grid (hidden_dim x dropout x lr_strategy)
    keys = list(GRID_SEARCH_SPACE.keys())
    values = list(GRID_SEARCH_SPACE.values())
    all_combinations = []
    for v in product(*values):
        combo = dict(zip(keys, v))
        combo["num_layers"] = args.num_layers
        combo["heads"] = args.heads
        all_combinations.append(combo)
    
    if args.grid_search:
        return all_combinations
    
    # Sample uniformly at random from grid
    rng = random.Random(args.seed)
    sampled = rng.sample(all_combinations, min(args.num_trials, len(all_combinations)))
    return sampled

# =====================================================================
# 4. MAIN TUNING PIPELINE
# =====================================================================

def main() -> None:
    args = parse_args()
    
    if args.patch_size is None:
        args.patch_size = 16 if args.backbone == "retfound" else 32

    if args.split_index >= args.split_total:
        raise ValueError("--split-index must be strictly less than --split-total")

    root = args.root.expanduser().resolve()
    prepared_dir = args.prepared_dir.expanduser().resolve() if args.prepared_dir else (root / "prepared")

    # Resolve target selection
    raw_target = None
    if args.singlelabel_mf:
        raw_target = "myopic_fundus"
    elif args.singlelabel_dr:
        raw_target = "diabetic_retinopathy"
    elif args.multilabel:
        raw_target = "multilabel"
    elif args.singlelabel:
        raw_target = args.singlelabel
    elif args.target:
        raw_target = args.target
    elif args.mode:
        raw_target = args.mode

    if args.labels:
        label_columns = [LABEL_ALIASES.get(lbl.lower(), lbl) for lbl in args.labels]
        is_multilabel = len(label_columns) > 1
        if is_multilabel:
            target_mode = "multilabel"
            short_name = "Multilabel"
        else:
            canonical_name = label_columns[0]
            short_name = LABEL_SHORT_NAMES.get(canonical_name, canonical_name.replace("_", " ").title())
            tag_code = LABEL_SHORT_NAMES.get(canonical_name, canonical_name.split("_")[0]).lower()
            target_mode = f"singlelabel_{tag_code}"
    elif raw_target is not None:
        raw_lower = raw_target.lower().strip()
        if raw_lower in ["multi", "multilabel", "all"]:
            label_columns = ALL_MULTILABEL_LABELS
            is_multilabel = True
            target_mode = "multilabel"
            short_name = "Multilabel"
        else:
            cleaned = raw_lower.replace("singlelabel_", "")
            canonical_name = LABEL_ALIASES.get(cleaned, cleaned)
            label_columns = [canonical_name]
            is_multilabel = False
            short_name = LABEL_SHORT_NAMES.get(canonical_name, canonical_name.replace("_", " ").title())
            tag_code = LABEL_SHORT_NAMES.get(canonical_name, canonical_name.split("_")[0]).lower()
            target_mode = f"singlelabel_{tag_code}"
    else:
        # Default: single-label Diabetic Retinopathy
        canonical_name = "diabetic_retinopathy"
        label_columns = [canonical_name]
        is_multilabel = False
        short_name = "DR"
        target_mode = "singlelabel_dr"

    # Configure defaults based on mode
    if is_multilabel:
        default_project = "Hyperparam-Tuning-Multilabel-ImgOnly"
        default_group = "Multilabel-ImgOnly-Tuning"
        default_prefix = "hparam_multi"
        default_tags = ["hparam_tuning", "multilabel", "image_only"]
        default_out_dir_name = "hparam_tuning_multilabel_img_only"
        experiment_type = "multilabel_img_only"
    else:
        tag_code = LABEL_SHORT_NAMES.get(label_columns[0], label_columns[0].split("_")[0]).lower()
        default_project = f"Hyperparam-Tuning-Singlelabel-{short_name}"
        default_group = f"Singlelabel-{short_name}-Image-Only"
        default_prefix = f"hparam_{tag_code}"
        default_tags = ["hparam_tuning", f"singlelabel_{tag_code}", label_columns[0], "image_only"]
        default_out_dir_name = f"hparam_tuning_singlelabel_{tag_code}"
        experiment_type = f"singlelabel_{tag_code}_img_only"

    wandb_project = args.wandb_project or default_project
    wandb_group = args.wandb_group or default_group
    wandb_prefix = args.wandb_run_name_prefix or default_prefix
    wandb_tags = args.wandb_tags if args.wandb_tags is not None else default_tags

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else (root / "runs" / default_out_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed, args.deterministic)
    device = select_device(args.device)
    amp_enabled = not args.no_amp and device.type == "cuda"
    
    print("===================================================================")
    print(f" HYPERPARAMETER TUNING: {short_name.upper()} Image-Only Model")
    print(f" Target Mode: {target_mode} | Labels ({len(label_columns)}): {label_columns}")
    print("===================================================================")
    print(f"Device: {device} | AMP: {amp_enabled} | Backbone: {args.backbone}")
    print(f"Dataset root: {root}")
    print(f"Output directory: {output_dir}\n")

    # Load DataFrames
    train_frame = pd.read_csv(prepared_dir / args.train_manifest)
    val_frame = pd.read_csv(prepared_dir / args.validation_manifest)
    
    # Extract top 20 comorbidities
    train_frame, top_comorb = preprocess_comorbidities(train_frame, top_k=20)
    val_frame, _ = preprocess_comorbidities(val_frame, valid_comorb=top_comorb)

    # Infer demographic columns & specs
    demo_columns = infer_demographic_columns(train_frame, None, label_columns)
    demo_specs = fit_demographic_specs(train_frame, demo_columns)

    # Image Transforms
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

    # Singlelabel DR Image-Only Dataset kwargs
    dataset_kwargs = {
        "image_dir": root / "fundus_photos_512",
        "target_cols": label_columns,
        "demo_cols": demo_columns,
        "demo_specs": demo_specs,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "connectivity": args.connectivity,
        "prediction_level": "dual_hub",
        "train_frame": train_frame
    }

    train_dataset = BRSETGraphDataset(
        data=train_frame,
        validate_paths=True,
        transform=train_transform,
        **dataset_kwargs
    )
    val_dataset = BRSETGraphDataset(
        data=val_frame,
        validate_paths=False,
        transform=val_transform,
        **dataset_kwargs
    )

    train_targets = np.vstack([s["label_vector"] for s in train_dataset.samples])
    pos_weight = calculate_positive_weights(train_targets).to(device)

    pos_counts = np.maximum(train_targets.sum(axis=0), 1.0)
    neg_counts = np.maximum(train_targets.shape[0] - pos_counts, 1.0)
    adaptive_biases = np.log(pos_counts / neg_counts).astype(np.float32)

    print("-------------------------------------------------------------------")
    print(f" Dataset Target Statistics (Total Samples: {len(train_dataset.samples)}):")
    for idx, lbl in enumerate(label_columns):
        p_count = int(pos_counts[idx])
        pct = (p_count / len(train_dataset.samples)) * 100.0
        w = float(pos_weight[idx].item())
        bias = float(adaptive_biases[idx])
        print(f"   • [{lbl}] Positive: {p_count}/{len(train_dataset.samples)} ({pct:.2f}%) | Pos Weight: {w:.2f} | Prior Logit Bias: {bias:.2f}")
    print("-------------------------------------------------------------------\n")

    # Generate trials
    all_trials = generate_trial_configs(args)
    total_trials = len(all_trials)

    # Slice for parallel array jobs
    start_idx = total_trials * args.split_index // args.split_total
    end_idx = total_trials * (args.split_index + 1) // args.split_total
    my_trials = all_trials[start_idx:end_idx]

    print(f"Total Trial Configurations: {total_trials}")
    print(f"Worker Split {args.split_index + 1}/{args.split_total} - Processing trials {start_idx + 1} to {end_idx} ({len(my_trials)} trials).\n")

    # Trial Execution Loop
    for trial_idx, hparams in enumerate(my_trials, start=start_idx + 1):
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        h_dim = hparams["hidden_dim"]
        dropout = hparams["dropout"]
        lr_strategy = hparams.get("lr_strategy", "uniform")
        n_layers = hparams.get("num_layers", args.num_layers)
        heads = hparams.get("heads", args.heads)
        
        # Adaptive micro-batching for large models to prevent CUDA OOM
        if h_dim >= 512:
            micro_batch_size = max(1, args.batch_size // 2)
            accum_steps = max(1, args.batch_size // micro_batch_size)
        else:
            micro_batch_size = args.batch_size
            accum_steps = 1

        trial_name = f"trial_{trial_idx:02d}_h{h_dim}_drop{dropout}_{lr_strategy}_lr"
        print("===================================================================")
        print(f"=== [TRIAL {trial_idx}/{total_trials}] {trial_name.upper()} ===")
        print(f"=== Params: h_dim={h_dim}, dropout={dropout}, lr_strategy={lr_strategy} ===")
        print(f"=== Fixed: num_layers={n_layers}, heads={heads} ===")
        print(f"=== Micro-batch: {micro_batch_size}, Grad Accumulation: {accum_steps} ===")
        print("===================================================================\n")

        run_dir = output_dir / trial_name
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "final_checkpoint.pt"

        train_loader = create_dataloader(
            dataset=train_dataset, batch_size=micro_batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            seed=args.seed + trial_idx, is_graph=True
        )
        val_loader = create_dataloader(
            dataset=val_dataset, batch_size=micro_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            seed=args.seed, is_graph=True
        )

        model = PatientGraphModel(
            num_labels=len(label_columns),
            demographic_specs=demo_specs,
            hidden_dim=h_dim,
            num_layers=n_layers,
            dropout=dropout,
            heads=heads,
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

        gnn_lr = args.learning_rate
        cnn_lr = args.cnn_learning_rate

        if lr_strategy == "split" and any(p.requires_grad for p in model.cnn.parameters()):
            cnn_params = [p for p in model.cnn.parameters() if p.requires_grad]
            cnn_param_ids = set(id(p) for p in cnn_params)
            gnn_params = [p for p in model.parameters() if p.requires_grad and id(p) not in cnn_param_ids]
            optimizer = torch.optim.AdamW([
                {"params": gnn_params, "lr": gnn_lr, "name": "gnn"},
                {"params": cnn_params, "lr": cnn_lr, "name": "cnn"},
            ])
            print(f"=== Optimizer: SPLIT LR (GNN={gnn_lr}, CNN Backbone={cnn_lr}) ===")
        else:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=gnn_lr)
            print(f"=== Optimizer: UNIFORM LR (All trainable={gnn_lr}) ===")

        scaler = make_grad_scaler(amp_enabled)
        
        warmup_epochs = max(1, int(0.1 * args.epochs))
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)

        trial_config = {
            "experiment": experiment_type,
            "mode": target_mode,
            "labels": label_columns,
            "no_context": True,
            "no_image": False,
            "trial_idx": trial_idx,
            "trial_name": trial_name,
            "hidden_dim": h_dim,
            "num_layers": n_layers,
            "heads": heads,
            "dropout": dropout,
            "lr_strategy": lr_strategy,
            "learning_rate": gnn_lr,
            "cnn_learning_rate": cnn_lr if lr_strategy == "split" else gnn_lr,
            "batch_size": args.batch_size,
            "micro_batch_size": micro_batch_size,
            "accum_steps": accum_steps,
            "seed": args.seed,
            "demo_cols": demo_columns,
            "demo_specs": [ds.__dict__ for ds in demo_specs],
            "train_manifest": args.train_manifest,
            "backbone": args.backbone,
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "connectivity": args.connectivity
        }
        write_json(run_dir / "run_config.json", trial_config)

        run_name = f"{wandb_prefix}-{trial_name}"
        wandb_run = start_wandb_run(
            project=wandb_project,
            run_name=run_name,
            config=trial_config,
            output_dir=run_dir,
            entity=args.wandb_entity,
            group=wandb_group,
            tags=wandb_tags,
            mode=args.wandb_mode
        )

        history = []
        best_metrics = {
            "macro_auprc": -1.0,
            "macro_auroc": -1.0,
            "macro_f1_calibrated": -1.0,
            "min_val_loss": float("inf")
        }

        for epoch in range(1, args.epochs + 1):
            train_res = graph_epoch(
                model=model, loader=train_loader, criterion=criterion, device=device,
                amp_enabled=amp_enabled, label_columns=label_columns, optimizer=optimizer,
                scaler=scaler, desc=f"[{trial_name}] E{epoch:02d} Train", threshold=args.threshold,
                accum_steps=accum_steps
            )
            scheduler.step()

            val_res = graph_epoch(
                model=model, loader=val_loader, criterion=criterion, device=device,
                amp_enabled=amp_enabled, label_columns=label_columns, optimizer=None,
                scaler=None, desc=f"[{trial_name}] E{epoch:02d} Val", threshold=args.threshold
            )

            best_thresholds, epoch_threshold_df = find_optimal_thresholds(
                targets=val_res["targets"],
                probabilities=val_res["probabilities"],
                label_names=label_columns,
                step=0.05
            )
            epoch_calibrated_f1 = epoch_threshold_df["validation_f1"].mean()
            epoch_balanced_acc = epoch_threshold_df["validation_balanced_accuracy"].mean()

            calibrated_predictions = (val_res["probabilities"] >= best_thresholds.reshape(1, -1)).astype(int)
            epoch_calibrated_micro_f1 = f1_score(val_res["targets"], calibrated_predictions, average="micro", zero_division=0)

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
            current_macro_auroc = val_res["summary"]["macro_auroc"]
            current_val_loss = val_res["loss"]

            raw_model = getattr(model, "_orig_mod", model)
            cnn_has_trainable = hasattr(raw_model, "cnn") and any(p.requires_grad for p in raw_model.cnn.parameters())
            if args.backbone == "retfound" and not cnn_has_trainable:
                state_dict_to_save = {k: v for k, v in raw_model.state_dict().items() if not k.startswith("cnn.")}
            else:
                state_dict_to_save = raw_model.state_dict()

            checkpoint_data = {
                "model_state_dict": state_dict_to_save,
                "hyperparams": trial_config,
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
                    wandb_run.summary["best_macro_f1_calibrated"] = epoch_calibrated_f1

            if current_val_loss < best_metrics["min_val_loss"]:
                best_metrics["min_val_loss"] = current_val_loss
                if wandb_run is not None:
                    wandb_run.summary["best_val_loss"] = current_val_loss

            if wandb_run is not None:
                y_true = val_res["targets"]
                y_prob = val_res["probabilities"]

                lrs = scheduler.get_last_lr()
                log_dict = {
                    "epoch": epoch,
                    "lr": lrs[0],
                    "lr/gnn": lrs[0],
                    "lr/cnn": lrs[1] if len(lrs) > 1 else lrs[0],
                    "train/loss": train_res["loss"],
                    "val/loss": val_res["loss"],
                    "val/macro_f1_uncalibrated": val_res["summary"]["macro_f1"],
                    "val/micro_f1_uncalibrated": val_res["summary"]["micro_f1"],
                    "val/macro_f1_calibrated": epoch_calibrated_f1,
                    "val/micro_f1_calibrated": epoch_calibrated_micro_f1,
                    "val/macro_auroc": val_res["summary"]["macro_auroc"],
                    "val/macro_auprc": val_res["summary"]["macro_auprc"],
                    "val/macro_accuracy_calibrated": epoch_threshold_df["validation_accuracy"].mean(),
                    "val/macro_balanced_acc_calibrated": epoch_balanced_acc,
                }

                uncalibrated_preds = (y_prob >= 0.5).astype(int)
                calibrated_preds = (y_prob >= best_thresholds.reshape(1, -1)).astype(int)
                for idx, label in enumerate(label_columns):
                    prefix = f"{idx+1:02d}_"
                    try:
                        auroc = roc_auc_score(y_true[:, idx], y_prob[:, idx])
                    except ValueError:
                        auroc = 0.0
                    
                    try:
                        auprc = average_precision_score(y_true[:, idx], y_prob[:, idx])
                    except ValueError:
                        auprc = 0.0

                    calib_f1_normal = f1_score(y_true[:, idx], calibrated_preds[:, idx], average="binary", zero_division=0)
                    uncalib_f1_normal = f1_score(y_true[:, idx], uncalibrated_preds[:, idx], average="binary", zero_division=0)

                    log_dict[f"val_auroc_per_class/{prefix}{label}"] = auroc
                    log_dict[f"val_auprc_per_class/{prefix}{label}"] = auprc
                    log_dict[f"val_calibrated_f1_per_class/{prefix}{label}"] = calib_f1_normal
                    log_dict[f"val_uncalibrated_f1_per_class/{prefix}{label}"] = uncalib_f1_normal
                    log_dict[f"val_optimal_threshold/{prefix}{label}"] = epoch_threshold_df.iloc[idx]["optimal_threshold"]

                wandb.log(log_dict, step=epoch)

        # Load best AUPRC checkpoint and rerun full evaluation on validation set
        best_auprc_path = run_dir / "checkpoint_best_macro_auprc.pt"
        if best_auprc_path.exists():
            ckpt = torch.load(best_auprc_path, map_location=device)
            raw_model = getattr(model, "_orig_mod", model)
            raw_model.load_state_dict(ckpt["model_state_dict"], strict=False)
            print(f"\n--> Loaded best AUPRC checkpoint from epoch {ckpt.get('epoch', 'N/A')}. Re-running validation evaluation...")

        final_val_res = graph_epoch(
            model=model, loader=val_loader, criterion=criterion, device=device,
            amp_enabled=amp_enabled, label_columns=label_columns, optimizer=None,
            scaler=None, desc=f"[{trial_name}] Final Best Val Eval", threshold=args.threshold
        )

        best_thresholds, final_threshold_df = find_optimal_thresholds(
            targets=final_val_res["targets"],
            probabilities=final_val_res["probabilities"],
            label_names=label_columns,
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
            "final_val_macro_f1_uncalibrated": float(final_val_res["summary"]["macro_f1"]),
            "final_val_micro_f1_uncalibrated": float(final_val_res["summary"]["micro_f1"]),
            "final_val_accuracy_calibrated": final_calibrated_acc,
            "final_val_balanced_acc_calibrated": final_balanced_acc,
            "per_class": {}
        }

        y_true = final_val_res["targets"]
        y_prob = final_val_res["probabilities"]
        for idx, label in enumerate(label_columns):
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
            uncalib_f1 = float(f1_score(y_true[:, idx], (y_prob[:, idx] >= 0.5).astype(int), average="binary", zero_division=0))
            opt_thresh = float(final_threshold_df.iloc[idx]["optimal_threshold"])

            final_eval_summary["per_class"][label] = {
                "auprc": auprc,
                "auroc": auroc,
                "calibrated_f1": calib_f1,
                "uncalibrated_f1": uncalib_f1,
                "optimal_threshold": opt_thresh
            }

        print("\n" + "=" * 65)
        print(f" TRIAL {trial_idx}/{total_trials} COMPLETED: {trial_name.upper()}")
        print(f" Best Checkpoint (by AUPRC) from Epoch {final_eval_summary['best_epoch']}:")
        print(f"   • Validation Macro AUPRC:         {final_eval_summary['final_val_macro_auprc']:.4f}")
        print(f"   • Validation Macro AUROC:         {final_eval_summary['final_val_macro_auroc']:.4f}")
        print(f"   • Validation Calibrated Macro F1: {final_eval_summary['final_val_macro_f1_calibrated']:.4f}")
        print(f"   • Validation Calibrated Micro F1: {final_eval_summary['final_val_micro_f1_calibrated']:.4f}")
        print(f"   • Validation Loss:                 {final_eval_summary['final_val_loss']:.4f}")
        print("=" * 65 + "\n")

        write_json(run_dir / "best_val_eval_results.json", final_eval_summary)

        if wandb_run is not None:
            # Save final best-model validation scores directly to wandb.run.summary for clean hyperparameter comparison
            wandb_run.summary["best_val/epoch"] = final_eval_summary["best_epoch"]
            wandb_run.summary["best_val/loss"] = final_eval_summary["final_val_loss"]
            wandb_run.summary["best_val/macro_auprc"] = final_eval_summary["final_val_macro_auprc"]
            wandb_run.summary["best_val/macro_auroc"] = final_eval_summary["final_val_macro_auroc"]
            wandb_run.summary["best_val/macro_f1_calibrated"] = final_eval_summary["final_val_macro_f1_calibrated"]
            wandb_run.summary["best_val/micro_f1_calibrated"] = final_eval_summary["final_val_micro_f1_calibrated"]
            wandb_run.summary["best_val/macro_accuracy_calibrated"] = final_eval_summary["final_val_accuracy_calibrated"]
            wandb_run.summary["best_val/macro_balanced_acc_calibrated"] = final_eval_summary["final_val_balanced_acc_calibrated"]

            for idx, label in enumerate(label_columns):
                prefix = f"{idx+1:02d}_"
                c_stats = final_eval_summary["per_class"][label]
                wandb_run.summary[f"best_val_per_class/auprc_{prefix}{label}"] = c_stats["auprc"]
                wandb_run.summary[f"best_val_per_class/auroc_{prefix}{label}"] = c_stats["auroc"]
            wandb_run.finish()

        del model, optimizer, train_loader, val_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nHyperparameter tuning complete!")


if __name__ == "__main__":
    main()
