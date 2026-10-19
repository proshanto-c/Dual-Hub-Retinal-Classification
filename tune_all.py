#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import re
from collections import Counter
from torchvision import transforms
from torchvision.transforms import InterpolationMode

try:
    import wandb
except ImportError:
    wandb = None

# --- SUPPRESS TERMINAL WARNINGS ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", message=".*An issue occurred while importing 'torch-scatter'.*")
warnings.filterwarnings("ignore", message=".*An issue occurred while importing 'torch-sparse'.*")

# --- MODULAR IMPORTS ---
from data_functions.BRSETGraphDataset import BRSETGraphDataset
from gnn.patientgraphmodel import PatientGraphModel, DemographicSpec
from utils.metrics import find_optimal_thresholds

from utils.util_functions import (
    set_seed, 
    select_device, 
    print_runtime, 
    write_json, 
    make_grad_scaler
)
from utils.data_functions import calculate_positive_weights, create_dataloader
from utils.train_functions import graph_epoch
from utils.metrics import save_predictions
from utils.logger import start_wandb_run, finish_wandb_run
from utils.feature_engineering import (
    preprocess_comorbidities, 
    infer_demographic_columns, 
    fit_demographic_specs, 
    DEFAULT_DEMOGRAPHIC_CANDIDATES
)
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

# =====================================================================
# 1. METADATA & DEMOGRAPHICS CONFIGURATION
# =====================================================================

ALL_LABELS = [
    "increased_cup_disc", "drusens", "diabetic_retinopathy", "macular_edema",
    "scar", "amd", "myopic_fundus"
]



# =====================================================================
# 2. ORCHESTRATION ARGUMENTS
# =====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune GNN hyperparameters across 39 model configurations.")
    p.add_argument("--root", type=Path, required=True, help="Dataset root directory.")
    p.add_argument("--prepared-dir", type=Path, default=None, help="Directory containing train.csv/val.csv.")
    
    p.add_argument("--train-manifest", type=str, default="image_train_12.csv")
    p.add_argument("--validation-manifest", type=str, default="image_validation_12.csv")
    
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--backbone", type=str, choices=["retfound", "resnet"], default="resnet")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--patch-size", type=int, default=None, help="Defaults to 16 for retfound, 32 for resnet")
    p.add_argument("--connectivity", default="fc", help="Patch graph connectivity (fc, 4, or 8)")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--monitor", choices=["val_loss", "val_macro_f1"], default="val_loss")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--compile", action="store_true", help="Compile model with torch.compile (disabled by default due to PyG HeteroConv recompilations).")
    p.add_argument("--num-trials", type=int, default=1, help="Number of random search trials per model config.")

    # Splitting Arguments
    p.add_argument("--split-total", type=int, default=1, help="Total number of parallel jobs (e.g. 2).")
    p.add_argument("--split-index", type=int, default=0, help="The 0-indexed chunk to run (e.g. 0 or 1).")

    # W&B Arguments
    p.add_argument("--wandb-project", type=str, default="GNN-Tuning-27-Models")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name-prefix", type=str, default="tune")
    p.add_argument("--wandb-group", type=str, default=None)
    p.add_argument("--wandb-tags", nargs="*", default=None)
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])

    return p.parse_args()


# =====================================================================
# 3. MAIN TUNING PIPELINE
# =====================================================================

def main() -> None:
    args = parse_args()
    
    if args.split_index >= args.split_total:
        raise ValueError("--split-index must be strictly less than --split-total")
        
    root = args.root.expanduser().resolve()
    prepared_dir = args.prepared_dir.expanduser().resolve() if args.prepared_dir else (root / "prepared")
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else (root / "runs" / "tuning_experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Environment Setup
    if args.patch_size is None:
        args.patch_size = 16 if args.backbone == "retfound" else 32
        
    set_seed(args.seed, args.deterministic)
    device = select_device(args.device)
    print_runtime(device)
    amp_enabled = device.type == "cuda" and not args.no_amp

    # 2. Metadata Extraction & Preprocessing
    train_frame = pd.read_csv(prepared_dir / args.train_manifest)
    val_frame = pd.read_csv(prepared_dir / args.validation_manifest)
    
    train_frame, top_comorb = preprocess_comorbidities(train_frame, top_k=20)
    val_frame, _ = preprocess_comorbidities(val_frame, valid_comorb=top_comorb)
    
    dynamic_comorb_cols = [f"comorbidity_{c.replace(' ', '_')}" for c in top_comorb]
    DEFAULT_DEMOGRAPHIC_CANDIDATES.extend(dynamic_comorb_cols)
    
    demo_columns = infer_demographic_columns(train_frame, None, ALL_LABELS)
    demo_specs = fit_demographic_specs(train_frame, demo_columns)

    # Transforms
    norm_step = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])] if args.backbone == "resnet" else []

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        *norm_step
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        *norm_step
    ])
    
    # 3. Define the Model Configurations
    models_list = [
        {"name": "multilabel_img_ctx", "group": "Multilabel", "labels": ALL_LABELS, "no_context": False, "no_image": False},
        {"name": "multilabel_img_only", "group": "Multilabel", "labels": ALL_LABELS, "no_context": True, "no_image": False},
        {"name": "multilabel_ctx_only", "group": "Multilabel", "labels": ALL_LABELS, "no_context": False, "no_image": True},
    ]
    
    for label in ALL_LABELS:
        group_name = label.replace('_', ' ').title()
        models_list.append({"name": f"singlelabel_{label}_img_ctx", "group": group_name, "labels": [label], "no_context": False, "no_image": False})
        models_list.append({"name": f"singlelabel_{label}_img_only", "group": group_name, "labels": [label], "no_context": True, "no_image": False})
        models_list.append({"name": f"singlelabel_{label}_ctx_only", "group": group_name, "labels": [label], "no_context": False, "no_image": True})

    # 4. Filter for Splitting Across Devices
    total_models = len(models_list)
    start_idx = total_models * args.split_index // args.split_total
    end_idx = total_models * (args.split_index + 1) // args.split_total
    my_models = models_list[start_idx:end_idx]

    print(f"\nTotal Configurations Available: {total_models}")
    print(f"Device Split {args.split_index + 1}/{args.split_total} - Processing configs {start_idx} to {end_idx - 1} ({len(my_models)} total configs).")

    # 5. Configuration Loop
    for exp_idx, exp in enumerate(my_models, start=1):
        print(f"\n===================================================================")
        print(f"=== [CONFIG {exp_idx}/{len(my_models)}] {exp['name'].upper()} ===")
        print(f"===================================================================\n")
        
        label_columns = exp["labels"]
        
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

        # Initialize dataset once per configuration
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
        
        # --- Inter-Label Distribution Weights ---
        max_pos = float(pos_counts.max())
        class_weights = np.sqrt(max_pos / pos_counts).astype(np.float32)
        class_weights = torch.tensor(class_weights, device=device)
        
        config_dir = output_dir / exp["name"]
        config_dir.mkdir(parents=True, exist_ok=True)

        # Tuning Loop
        for trial in range(1, args.num_trials + 1):
            h_dim = 256
            n_layers = 3
            dropout = 0.25
            batch_size = 32
            learning_rate = 1e-4

            print(f"\n--- {exp['name']} | FINAL TRAINING ---")
            print(f"Params: h_dim={h_dim}, n_layers={n_layers}, drop={dropout:.4f}, batch={batch_size}, lr={learning_rate}")

            train_loader = create_dataloader(
                dataset=train_dataset, batch_size=batch_size, shuffle=True, 
                num_workers=args.num_workers, pin_memory=(device.type == "cuda"), 
                seed=args.seed + trial, is_graph=True
            )
            val_loader = create_dataloader(
                dataset=val_dataset, batch_size=batch_size, shuffle=False, 
                num_workers=args.num_workers, pin_memory=(device.type == "cuda"), 
                seed=args.seed, is_graph=True
            )

            model = PatientGraphModel(
                num_labels=len(label_columns), demographic_specs=demo_specs,
                hidden_dim=h_dim, num_layers=n_layers, dropout=dropout,
                prediction_level="dual_hub", initial_biases=adaptive_biases,
                use_context=not exp["no_context"],
                use_image_features=not exp.get("no_image", False),
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

            # Removed weight=class_weights to prevent massive double-weighted gradient spikes from rare labels!
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            scaler = make_grad_scaler(amp_enabled)
            # Linear warm-up from 1% of base LR up to 100% over the first 10% of epochs
            warmup_epochs = max(1, int(0.1 * args.epochs))
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)

            run_dir = config_dir / f"trial_{trial}_h{h_dim}_l{n_layers}"
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = run_dir / "final_checkpoint.pt"
            
            trial_config = {
                "experiment": exp["name"], "labels": label_columns, 
                "no_context": exp["no_context"], "no_image": exp.get("no_image", False),
                "trial": trial, "h_dim": h_dim, "n_layers": n_layers, "dropout": dropout,
                "batch_size": batch_size, "lr": learning_rate, "seed": args.seed,
                "demo_cols": demo_columns,
                "demo_specs": [ds.__dict__ for ds in demo_specs],
                "train_manifest": args.train_manifest,
                "backbone": args.backbone
            }
            write_json(run_dir / "run_config.json", trial_config)

            run_name = f"{args.wandb_run_name_prefix}-{exp['name']}-T{trial}"
            wandb_run = start_wandb_run(
                project=args.wandb_project, run_name=run_name, config=trial_config, output_dir=run_dir,
                entity=args.wandb_entity, group=exp['group'], tags=args.wandb_tags, mode=args.wandb_mode
            )

            # Dynamic checkpoint key
            checkpoint_key = "auprc" if len(label_columns) == 1 else "macro_auprc"
            checkpoint_key_f1 = "f1" if len(label_columns) == 1 else "micro_f1"
            
            history = []
            best_metrics = {
                checkpoint_key: -1.0,
                checkpoint_key_f1: -1.0
            }
            
            for epoch in range(1, args.epochs + 1):
                train_res = graph_epoch(
                    model=model, loader=train_loader, criterion=criterion, device=device, 
                    amp_enabled=amp_enabled, label_columns=label_columns, optimizer=optimizer, 
                    scaler=scaler, desc=f"[{exp['name']} T{trial}] E{epoch:02d} Train", threshold=args.threshold
                )
                scheduler.step()
                
                val_res = graph_epoch(
                    model=model, loader=val_loader, criterion=criterion, device=device, 
                    amp_enabled=amp_enabled, label_columns=label_columns, optimizer=None, 
                    scaler=None, desc=f"[{exp['name']} T{trial}] E{epoch:02d} Val", threshold=args.threshold
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
                
                # Checkpoints
                current_metrics = {
                    checkpoint_key: val_res["summary"]["macro_auprc"],
                    checkpoint_key_f1: epoch_calibrated_micro_f1
                }
                
                raw_model = getattr(model, "_orig_mod", model)
                if args.backbone == "retfound":
                    # RETFound backbone is frozen and not tuned, so omit static weights (cnn.) to save ~1.2GB per checkpoint
                    state_dict_to_save = {k: v for k, v in raw_model.state_dict().items() if not k.startswith("cnn.")}
                else:
                    # Tuned backbones (e.g. resnet) and all tuned model layers are saved in full
                    state_dict_to_save = raw_model.state_dict()
                    
                checkpoint_data = {
                    "model_state_dict": state_dict_to_save,
                    "hyperparams": trial_config,
                    "epoch": epoch
                }
                torch.save(checkpoint_data, checkpoint_path) # latest checkpoint
                
                for metric_name, val in current_metrics.items():
                    if val > best_metrics[metric_name]:
                        best_metrics[metric_name] = val
                        torch.save(checkpoint_data, run_dir / f"checkpoint_best_{metric_name}.pt")
                        
                        if wandb_run is not None:
                            wandb_run.summary[f"best_{metric_name}_checkpoint/epoch"] = epoch
                            wandb_run.summary[f"best_{metric_name}_checkpoint/val"] = val

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
                        "val/macro_accuracy_calibrated": epoch_threshold_df["validation_accuracy"].mean(),
                        "val/macro_balanced_acc_calibrated": epoch_balanced_acc,
                    }

                    uncalibrated_preds = (y_prob >= 0.5).astype(int)
                    calibrated_preds = (y_prob >= best_thresholds.reshape(1, -1)).astype(int)
                    for idx, label in enumerate(label_columns):
                        # Use the global index from ALL_LABELS so the prefix is identical across all models
                        global_idx = ALL_LABELS.index(label)
                        prefix = f"{global_idx+1:02d}_"
                        try:
                            auroc = roc_auc_score(y_true[:, idx], y_prob[:, idx])
                        except ValueError:
                            auroc = 0.0 
                        
                        try:
                            auprc = average_precision_score(y_true[:, idx], y_prob[:, idx])
                        except ValueError:
                            auprc = 0.0
                        
                        uncalib_f1_normal = f1_score(y_true[:, idx], uncalibrated_preds[:, idx], average="binary", zero_division=0)
                        uncalib_f1_macro = f1_score(y_true[:, idx], uncalibrated_preds[:, idx], average="macro", zero_division=0)
                        uncalib_f1_micro = f1_score(y_true[:, idx], uncalibrated_preds[:, idx], average="micro", zero_division=0)
                        
                        calib_f1_normal = f1_score(y_true[:, idx], calibrated_preds[:, idx], average="binary", zero_division=0)
                        calib_f1_macro = f1_score(y_true[:, idx], calibrated_preds[:, idx], average="macro", zero_division=0)
                        calib_f1_micro = f1_score(y_true[:, idx], calibrated_preds[:, idx], average="micro", zero_division=0)
                        
                        log_dict[f"val_auroc_per_class/{prefix}{label}"] = auroc
                        log_dict[f"val_auprc_per_class/{prefix}{label}"] = auprc
                        
                        # The optimal threshold used by the model (Youden for multilabel, F1 for single)
                        log_dict[f"val_optimal_threshold/{prefix}{label}"] = epoch_threshold_df.iloc[idx]["optimal_threshold"]
                        
                        # Explicit Calibration Tracking
                        log_dict[f"val_optimal_f1_threshold/{prefix}{label}"] = epoch_threshold_df.iloc[idx]["optimal_f1_threshold"]
                        log_dict[f"val_calibrated_f1_at_f1_thresh/{prefix}{label}"] = epoch_threshold_df.iloc[idx]["validation_f1_at_f1_thresh"]
                        if len(label_columns) > 1:
                            log_dict[f"val_optimal_youden_threshold/{prefix}{label}"] = epoch_threshold_df.iloc[idx]["optimal_youden_threshold"]
                            log_dict[f"val_calibrated_f1_at_youden_thresh/{prefix}{label}"] = epoch_threshold_df.iloc[idx]["validation_f1_at_youden_thresh"]
                        
                        log_dict[f"val_calibrated_f1_per_class/{prefix}{label}"] = calib_f1_normal
                        log_dict[f"val_calibrated_f1_macro_per_class/{prefix}{label}"] = calib_f1_macro
                        log_dict[f"val_calibrated_f1_micro_per_class/{prefix}{label}"] = calib_f1_micro
                        
                        log_dict[f"val_uncalibrated_f1_per_class/{prefix}{label}"] = uncalib_f1_normal
                        log_dict[f"val_uncalibrated_f1_macro_per_class/{prefix}{label}"] = uncalib_f1_macro
                        log_dict[f"val_uncalibrated_f1_micro_per_class/{prefix}{label}"] = uncalib_f1_micro
                        
                        log_dict[f"val_accuracy_per_class/{prefix}{label}"] = epoch_threshold_df.iloc[idx]["validation_accuracy"]
                        log_dict[f"val_balanced_acc_per_class/{prefix}{label}"] = epoch_threshold_df.iloc[idx]["validation_balanced_accuracy"]

                    wandb.log(log_dict, step=epoch)

            best_thresholds, threshold_df = find_optimal_thresholds(
                targets=val_res["targets"],
                probabilities=val_res["probabilities"],
                label_names=label_columns,
                step=0.01
            )
            
            print(f"\n--- Calibrated Optimal Thresholds (Trial {trial}) ---")
            print(threshold_df.to_string(index=False))

            val_preds = (val_res["probabilities"] >= best_thresholds).astype(np.int8)
            calibrated_macro_f1 = threshold_df["validation_f1"].mean()

            entity_ids = [s["image_id"] for s in val_dataset.samples]
            
            save_predictions(
                path=run_dir / "validation_predictions.csv",
                entity_ids=entity_ids,
                id_column_name="image_id",
                label_names=label_columns,
                targets=val_res["targets"],
                probabilities=val_res["probabilities"],
                predictions=val_preds
            )
            
            summary_data = {
                "final_validation_metric": float(calibrated_macro_f1 if args.monitor == "val_macro_f1" else val_res["loss"]),
                "calibrated_macro_f1": float(calibrated_macro_f1),
                "calibrated_thresholds": best_thresholds.tolist(),
                "validation": val_res["summary"]
            }
            write_json(run_dir / "summary.json", summary_data)

            if wandb_run is not None:
                wandb_run.summary["final_validation_metric"] = summary_data["final_validation_metric"]
                wandb_run.summary["calibrated_macro_f1"] = float(calibrated_macro_f1)
                
                for _, row in threshold_df.iterrows():
                    label = row["label"]
                    wandb_run.summary[f"calibrated_f1/{label}"] = row["validation_f1"]
                    wandb_run.summary[f"optimal_threshold/{label}"] = row["optimal_threshold"]
                
                wandb.save(str(run_dir / "summary.json"), base_path=str(run_dir))
                wandb.save(str(checkpoint_path), base_path=str(run_dir))
                
            finish_wandb_run(wandb_run)

    print("\nOrchestrated Tuning Complete!")

if __name__ == "__main__":
    main()
