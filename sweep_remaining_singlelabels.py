import argparse
import gc
import json
import os
import sys
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
# 1. REMAINING LABELS & CANDIDATE HYPERPARAMETER CONFIGURATIONS
# =====================================================================

# The 5 remaining single labels after DR and MF
REMAINING_LABELS = [
    "increased_cup_disc",  # ICD / Glaucoma suspicion
    "drusens",             # Drusen
    "macular_edema",       # Macular Edema (DME)
    "scar",                # Chorioretinal scar
    "amd",                 # Age-related macular degeneration
]

LABEL_PRETTY_NAMES = {
    "increased_cup_disc": "Glaucoma / Increased Cup-to-Disc",
    "drusens": "Drusen",
    "macular_edema": "Macular Edema",
    "scar": "Chorioretinal Scar",
    "amd": "Age-Related Macular Degeneration (AMD)",
}

LABEL_ALIASES = {
    "icd": "increased_cup_disc",
    "glaucoma": "increased_cup_disc",
    "increased_cup_disc": "increased_cup_disc",
    "drusen": "drusens",
    "drusens": "drusens",
    "me": "macular_edema",
    "macular_edema": "macular_edema",
    "dme": "macular_edema",
    "scar": "scar",
    "amd": "amd",
}

# The two candidate hyperparameter sets for head-to-head comparison
CANDIDATE_CONFIGS = [
    {
        "id": "h128_d025",
        "name": "128h_0.25d",
        "part": 1,
        "hidden_dim": 128,
        "dropout": 0.25,
        "lr_strategy": "uniform",
        "learning_rate": 1e-4,
        "num_layers": 3,
        "heads": 2,
    },
    {
        "id": "h512_d015",
        "name": "512h_0.15d",
        "part": 2,
        "hidden_dim": 512,
        "dropout": 0.15,
        "lr_strategy": "uniform",
        "learning_rate": 1e-4,
        "num_layers": 3,
        "heads": 2,
    },
]

# =====================================================================
# 2. CLI ARGUMENT PARSER
# =====================================================================

def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Head-to-head final sweep for the 5 remaining single labels (Image-Only)."
    )
    p.add_argument("--root", type=Path, required=True, help="Dataset root directory.")
    p.add_argument("--prepared-dir", type=Path, default=None, help="Directory containing train.csv/val.csv.")
    p.add_argument("--train-manifest", type=str, default="image_train_12.csv")
    p.add_argument("--validation-manifest", type=str, default="image_validation_12.csv")
    p.add_argument("--output-dir", type=Path, default=None, help="Output root directory for sweep checkpoints.")

    # Sweep partitioning: Part 1 (128h/0.25d), Part 2 (512h/0.15d), or both
    p.add_argument(
        "--part",
        type=str,
        default="all",
        choices=["1", "2", "part1", "part2", "all", "both"],
        help="Part of sweep to run: '1'/'part1' (128h/0.25d), '2'/'part2' (512h/0.15d), or 'all'/'both' (default: all).",
    )
    p.add_argument(
        "--labels", "--label",
        nargs="+",
        default=None,
        help="Filter to run specific label(s) (e.g., --labels icd drusens me, or --label amd).",
    )

    # Model & Architecture fixed parameters
    p.add_argument("--backbone", type=str, choices=["retfound", "resnet"], default="resnet")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--patch-size", type=int, default=None, help="Defaults to 16 for retfound, 32 for resnet")
    p.add_argument("--connectivity", default="fc", help="Patch graph connectivity: 'fc' (recommended), 4, or 8.")
    p.add_argument("--epochs", type=int, default=150, help="Number of training epochs (default: 150).")
    p.add_argument("--batch-size", type=int, default=32, help="Effective batch size (default: 32).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--compile", action="store_true", help="Compile model with torch.compile (disabled by default).")

    # Splitting Arguments for SLURM Job Arrays
    p.add_argument("--split-total", type=int, default=1, help="Total number of parallel jobs.")
    p.add_argument("--split-index", type=int, default=0, help="0-indexed worker chunk to run.")

    # Unified W&B Project Arguments
    p.add_argument(
        "--wandb-project",
        type=str,
        default="Singlelabel-Image-Only-Final-Sweep",
        help="W&B project name (all runs log to this same project).",
    )
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])

    return p.parse_args(args)


# =====================================================================
# 3. RUN GENERATION & COMPARISON LOGIC
# =====================================================================

def build_run_list(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Builds the list of (label, config) runs to execute."""
    # 1. Filter labels
    if args.labels:
        target_labels = []
        for raw_item in args.labels:
            raw_clean = raw_item.lower().strip()
            canonical = LABEL_ALIASES.get(raw_clean, raw_clean)
            if canonical not in REMAINING_LABELS:
                raise ValueError(f"Label '{raw_item}' is not among remaining labels: {REMAINING_LABELS}")
            if canonical not in target_labels:
                target_labels.append(canonical)
    else:
        target_labels = list(REMAINING_LABELS)

    # 2. Filter candidate configs by part
    part_str = str(args.part).lower()
    if part_str in ["1", "part1"]:
        configs = [c for c in CANDIDATE_CONFIGS if c["part"] == 1]
    elif part_str in ["2", "part2"]:
        configs = [c for c in CANDIDATE_CONFIGS if c["part"] == 2]
    else:
        configs = list(CANDIDATE_CONFIGS)

    # 3. Create full cartesian list of runs
    planned_runs = []
    run_idx = 1
    for label in target_labels:
        for cfg in configs:
            planned_runs.append({
                "run_id": f"{label}_{cfg['id']}",
                "global_idx": run_idx,
                "label": label,
                "pretty_name": LABEL_PRETTY_NAMES.get(label, label),
                "config": cfg,
            })
            run_idx += 1

    return planned_runs


def check_and_save_comparison(label: str, output_dir: Path) -> Optional[Dict[str, Any]]:
    """Checks if both candidate configs for `label` have completed, and computes head-to-head comparison."""
    label_dir = output_dir / label
    results = {}

    for cfg in CANDIDATE_CONFIGS:
        cfg_id = cfg["id"]
        res_file = label_dir / cfg_id / "best_val_eval_results.json"
        if res_file.exists():
            try:
                with open(res_file, "r") as f:
                    results[cfg_id] = json.load(f)
            except Exception:
                pass

    if len(results) == 2:
        cfg1_res = results["h128_d025"]
        cfg2_res = results["h512_d015"]

        auprc_1 = cfg1_res.get("final_val_macro_auprc", 0.0)
        auprc_2 = cfg2_res.get("final_val_macro_auprc", 0.0)
        diff = auprc_2 - auprc_1

        if auprc_2 > auprc_1:
            winner = "h512_d015 (512h_0.15d)"
            winner_id = "h512_d015"
        elif auprc_1 > auprc_2:
            winner = "h128_d025 (128h_0.25d)"
            winner_id = "h128_d025"
        else:
            winner = "TIE"
            winner_id = "tie"

        comparison = {
            "label": label,
            "pretty_name": LABEL_PRETTY_NAMES.get(label, label),
            "winner": winner,
            "winner_id": winner_id,
            "auprc_diff_512_minus_128": diff,
            "candidates": {
                "h128_d025": {
                    "name": "128h_0.25d_uniform",
                    "best_epoch": cfg1_res.get("best_epoch"),
                    "macro_auprc": auprc_1,
                    "macro_auroc": cfg1_res.get("final_val_macro_auroc"),
                    "calibrated_f1": cfg1_res.get("final_val_macro_f1_calibrated"),
                    "uncalibrated_f1": cfg1_res.get("final_val_macro_f1_uncalibrated"),
                    "val_loss": cfg1_res.get("final_val_loss"),
                },
                "h512_d015": {
                    "name": "512h_0.15d_uniform",
                    "best_epoch": cfg2_res.get("best_epoch"),
                    "macro_auprc": auprc_2,
                    "macro_auroc": cfg2_res.get("final_val_macro_auroc"),
                    "calibrated_f1": cfg2_res.get("final_val_macro_f1_calibrated"),
                    "uncalibrated_f1": cfg2_res.get("final_val_macro_f1_uncalibrated"),
                    "val_loss": cfg2_res.get("final_val_loss"),
                }
            }
        }

        comp_path = label_dir / "head_to_head_comparison.json"
        write_json(comp_path, comparison)

        print("\n" + "=" * 78)
        print(f" HEAD-TO-HEAD COMPARISON COMPLETE FOR: {LABEL_PRETTY_NAMES.get(label, label).upper()}")
        print("=" * 78)
        print(f"   • Candidate 1 (128h_0.25d): Val AUPRC = {auprc_1:.4f} (Epoch {cfg1_res.get('best_epoch')})")
        print(f"   • Candidate 2 (512h_0.15d): Val AUPRC = {auprc_2:.4f} (Epoch {cfg2_res.get('best_epoch')})")
        print(f"   --> WINNER: {winner} (Delta AUPRC: {diff:+.4f})")
        print("=" * 78 + "\n")

        return comparison

    return None


def print_overall_summary(output_dir: Path) -> None:
    """Prints a consolidated summary table across all remaining labels."""
    all_comparisons = []
    for label in REMAINING_LABELS:
        comp_file = output_dir / label / "head_to_head_comparison.json"
        if comp_file.exists():
            try:
                with open(comp_file, "r") as f:
                    all_comparisons.append(json.load(f))
            except Exception:
                pass

    if not all_comparisons:
        return

    print("\n" + "#" * 86)
    print(" SUMMARY OF FINAL HEAD-TO-HEAD SWEEPS ACROSS ALL REMAINING SINGLE LABELS")
    print("#" * 86)
    print(f"{'Label':<25} {'128h_0.25d':<12} {'512h_0.15d':<12} {'Winner':<24} {'Delta AUPRC':<10}")
    print("-" * 86)

    summary_records = {}
    for comp in all_comparisons:
        lbl = comp["label"]
        c1 = comp["candidates"]["h128_d025"]["macro_auprc"]
        c2 = comp["candidates"]["h512_d015"]["macro_auprc"]
        winner = comp["winner"]
        delta = comp["auprc_diff_512_minus_128"]
        print(f"{lbl:<25} {c1:<12.4f} {c2:<12.4f} {winner:<24} {delta:+0.4f}")
        summary_records[lbl] = comp

    print("#" * 86 + "\n")
    write_json(output_dir / "final_sweep_summary.json", summary_records)


# =====================================================================
# 4. MAIN SWEEP RUNNER
# =====================================================================

def main() -> None:
    args = parse_args()

    if args.patch_size is None:
        args.patch_size = 16 if args.backbone == "retfound" else 32

    if args.split_index >= args.split_total:
        raise ValueError("--split-index must be strictly less than --split-total")

    root = args.root.expanduser().resolve()
    prepared_dir = args.prepared_dir.expanduser().resolve() if args.prepared_dir else (root / "prepared")
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else (root / "runs" / "final_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed, args.deterministic)
    device = select_device(args.device)
    amp_enabled = not args.no_amp and device.type == "cuda"

    # Generate full planned run list
    all_runs = build_run_list(args)
    total_planned = len(all_runs)

    # Slice for parallel array jobs
    start_idx = total_planned * args.split_index // args.split_total
    end_idx = total_planned * (args.split_index + 1) // args.split_total
    my_runs = all_runs[start_idx:end_idx]

    print("===================================================================")
    print(" FINAL SWEEP: HEAD-TO-HEAD COMPARISON ON 5 REMAINING SINGLE LABELS")
    print("===================================================================")
    print(f"Device: {device} | AMP: {amp_enabled} | Backbone: {args.backbone}")
    print(f"Dataset root: {root}")
    print(f"Output root directory: {output_dir}")
    print(f"W&B Unified Project: {args.wandb_project}")
    print(f"Total planned runs: {total_planned} | Mode/Part: {args.part}")
    print(f"Worker Split {args.split_index + 1}/{args.split_total} - Processing runs {start_idx + 1} to {end_idx} ({len(my_runs)} runs).\n")

    if len(my_runs) == 0:
        print("No runs allocated for this worker split. Exiting.")
        return

    # Load DataFrames
    train_frame = pd.read_csv(prepared_dir / args.train_manifest)
    val_frame = pd.read_csv(prepared_dir / args.validation_manifest)

    # Comorbidities & Demographic specs
    train_frame, top_comorb = preprocess_comorbidities(train_frame, top_k=20)
    val_frame, _ = preprocess_comorbidities(val_frame, valid_comorb=top_comorb)

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

    # Cache for datasets so we only recreate when target label changes
    current_label = None
    train_dataset = None
    val_dataset = None
    pos_weight = None
    adaptive_biases = None
    demo_specs = None
    demo_columns = None

    # Loop over assigned runs
    for run_meta in my_runs:
        label = run_meta["label"]
        pretty_name = run_meta["pretty_name"]
        cfg = run_meta["config"]
        run_id = run_meta["run_id"]
        global_idx = run_meta["global_idx"]

        h_dim = cfg["hidden_dim"]
        dropout = cfg["dropout"]
        lr_strategy = cfg["lr_strategy"]
        learning_rate = cfg["learning_rate"]
        n_layers = cfg["num_layers"]
        heads = cfg["heads"]
        part_num = cfg["part"]

        # Adaptive micro-batching for 512h to prevent CUDA OOM
        if h_dim >= 512:
            micro_batch_size = max(1, args.batch_size // 2)
            accum_steps = max(1, args.batch_size // micro_batch_size)
        else:
            micro_batch_size = args.batch_size
            accum_steps = 1

        print("===================================================================")
        print(f"=== [RUN {global_idx}/{total_planned}] PART {part_num}: {label.upper()} ({cfg['name'].upper()}) ===")
        print(f"=== Condition: {pretty_name} ===")
        print(f"=== Config: h_dim={h_dim}, dropout={dropout}, lr={learning_rate} (uniform) ===")
        print(f"=== Micro-batch: {micro_batch_size}, Grad Accumulation: {accum_steps} ===")
        print("===================================================================\n")

        # Re-initialize dataset if target label changed
        if label != current_label:
            del train_dataset, val_dataset
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            current_label = label
            label_columns = [label]

            demo_columns = infer_demographic_columns(train_frame, None, label_columns)
            demo_specs = fit_demographic_specs(train_frame, demo_columns)

            dataset_kwargs = {
                "image_dir": root / "fundus_photos_512",
                "target_cols": label_columns,
                "demo_cols": demo_columns,
                "demo_specs": demo_specs,
                "image_size": args.image_size,
                "patch_size": args.patch_size,
                "use_context": False,          # Strictly Image-Only
                "use_image_features": True,
                "prediction_level": "dual_hub",
                "patch_connectivity": args.connectivity,
            }

            train_dataset = BRSETGraphDataset(
                data=train_frame,
                validate_paths=False,
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

            p_count = int(pos_counts[0])
            total_samples = len(train_dataset.samples)
            pct = (p_count / total_samples) * 100.0
            print(f"Target Statistics for [{label}]: {p_count}/{total_samples} positives ({pct:.2f}%) | Pos Weight: {pos_weight[0].item():.2f} | Prior Bias: {adaptive_biases[0]:.2f}\n")

        # Dataloaders
        train_loader = create_dataloader(
            dataset=train_dataset, batch_size=micro_batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            seed=args.seed + global_idx, is_graph=True
        )
        val_loader = create_dataloader(
            dataset=val_dataset, batch_size=micro_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            seed=args.seed, is_graph=True
        )

        run_dir = output_dir / label / cfg["id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "final_checkpoint.pt"

        # Initialize Image-Only Model
        model = PatientGraphModel(
            num_labels=1,
            demographic_specs=demo_specs,
            hidden_dim=h_dim,
            num_layers=n_layers,
            dropout=dropout,
            heads=heads,
            prediction_level="dual_hub",
            initial_biases=adaptive_biases,
            use_context=False,          # Strictly Image-Only
            use_image_features=True,
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
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
        scaler = make_grad_scaler(amp_enabled)

        warmup_epochs = max(1, int(0.1 * args.epochs))
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)

        trial_config = {
            "experiment": f"final_sweep_singlelabel_{label}",
            "target": label,
            "label_pretty_name": pretty_name,
            "part": part_num,
            "candidate_id": cfg["id"],
            "candidate_name": cfg["name"],
            "no_context": True,
            "no_image": False,
            "hidden_dim": h_dim,
            "num_layers": n_layers,
            "heads": heads,
            "dropout": dropout,
            "lr_strategy": lr_strategy,
            "learning_rate": learning_rate,
            "batch_size": args.batch_size,
            "micro_batch_size": micro_batch_size,
            "accum_steps": accum_steps,
            "seed": args.seed,
            "epochs": args.epochs,
            "backbone": args.backbone,
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "connectivity": args.connectivity,
        }
        write_json(run_dir / "run_config.json", trial_config)

        # W&B: All runs log to same project, grouped by label for direct side-by-side comparison
        run_name = f"{label}_{cfg['id']}"
        wandb_run = start_wandb_run(
            project=args.wandb_project,
            run_name=run_name,
            config=trial_config,
            output_dir=run_dir,
            entity=args.wandb_entity,
            group=f"Singlelabel-{label}",
            tags=["final_sweep", "singlelabel", label, cfg["id"], f"part{part_num}"],
            mode=args.wandb_mode
        )

        history = []
        best_metrics = {
            "macro_auprc": -1.0,
            "macro_auroc": -1.0,
            "macro_f1_calibrated": -1.0,
            "min_val_loss": float("inf")
        }

        # Training Loop
        for epoch in range(1, args.epochs + 1):
            train_res = graph_epoch(
                model=model, loader=train_loader, criterion=criterion, device=device,
                amp_enabled=amp_enabled, label_columns=[label], optimizer=optimizer,
                scaler=scaler, desc=f"[{run_id}] E{epoch:02d} Train", threshold=args.threshold,
                accum_steps=accum_steps
            )
            scheduler.step()

            val_res = graph_epoch(
                model=model, loader=val_loader, criterion=criterion, device=device,
                amp_enabled=amp_enabled, label_columns=[label], optimizer=None,
                scaler=None, desc=f"[{run_id}] E{epoch:02d} Val", threshold=args.threshold
            )

            best_thresholds, epoch_threshold_df = find_optimal_thresholds(
                targets=val_res["targets"],
                probabilities=val_res["probabilities"],
                label_names=[label],
                step=0.05
            )
            epoch_calibrated_f1 = float(epoch_threshold_df["validation_f1"].mean())
            epoch_balanced_acc = float(epoch_threshold_df["validation_balanced_accuracy"].mean())

            calibrated_predictions = (val_res["probabilities"] >= best_thresholds.reshape(1, -1)).astype(int)
            epoch_calibrated_micro_f1 = float(f1_score(val_res["targets"], calibrated_predictions, average="micro", zero_division=0))

            current_macro_auprc = float(val_res["summary"]["macro_auprc"])
            current_macro_auroc = float(val_res["summary"]["macro_auroc"])
            current_val_loss = float(val_res["loss"])

            # Save best checkpoint based on AUPRC
            if current_macro_auprc > best_metrics["macro_auprc"]:
                best_metrics["macro_auprc"] = current_macro_auprc
                best_metrics["macro_auroc"] = current_macro_auroc
                best_metrics["macro_f1_calibrated"] = epoch_calibrated_f1
                best_metrics["min_val_loss"] = current_val_loss

                best_ckpt_path = run_dir / "checkpoint_best_macro_auprc.pt"
                raw_model = getattr(model, "_orig_mod", model)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "best_metrics": best_metrics,
                    "optimal_thresholds": best_thresholds.tolist(),
                    "hyperparameters": trial_config,
                }, best_ckpt_path)

            if wandb_run is not None:
                log_dict = {
                    "train/loss": train_res["loss"],
                    "val/loss": val_res["loss"],
                    "val/macro_auprc": current_macro_auprc,
                    "val/macro_auroc": current_macro_auroc,
                    "val/macro_f1_calibrated": epoch_calibrated_f1,
                    "val/macro_f1_uncalibrated": val_res["summary"]["macro_f1"],
                    "val/balanced_accuracy": epoch_balanced_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                wandb.log(log_dict, step=epoch)

        # Save final checkpoint
        raw_model = getattr(model, "_orig_mod", model)
        torch.save({
            "epoch": args.epochs,
            "model_state_dict": raw_model.state_dict(),
            "best_metrics": best_metrics,
            "hyperparameters": trial_config,
        }, checkpoint_path)

        # Re-evaluate best AUPRC checkpoint
        best_auprc_path = run_dir / "checkpoint_best_macro_auprc.pt"
        if best_auprc_path.exists():
            ckpt = torch.load(best_auprc_path, map_location=device)
            raw_model.load_state_dict(ckpt["model_state_dict"], strict=False)
            best_epoch_num = ckpt.get("epoch", args.epochs)
        else:
            best_epoch_num = args.epochs

        final_val_res = graph_epoch(
            model=model, loader=val_loader, criterion=criterion, device=device,
            amp_enabled=amp_enabled, label_columns=[label], optimizer=None,
            scaler=None, desc=f"[{run_id}] Final Best Val Eval", threshold=args.threshold
        )

        best_thresholds, final_threshold_df = find_optimal_thresholds(
            targets=final_val_res["targets"],
            probabilities=final_val_res["probabilities"],
            label_names=[label],
            step=0.05
        )

        final_calibrated_f1 = float(final_threshold_df["validation_f1"].mean())
        final_calibrated_acc = float(final_threshold_df["validation_accuracy"].mean())
        final_balanced_acc = float(final_threshold_df["validation_balanced_accuracy"].mean())
        final_calibrated_preds = (final_val_res["probabilities"] >= best_thresholds.reshape(1, -1)).astype(int)
        final_calibrated_micro_f1 = float(f1_score(final_val_res["targets"], final_calibrated_preds, average="micro", zero_division=0))

        y_true = final_val_res["targets"]
        y_prob = final_val_res["probabilities"]
        try:
            auroc = float(roc_auc_score(y_true[:, 0], y_prob[:, 0]))
        except ValueError:
            auroc = 0.0
        try:
            auprc = float(average_precision_score(y_true[:, 0], y_prob[:, 0]))
        except ValueError:
            auprc = 0.0

        calib_f1 = float(f1_score(y_true[:, 0], final_calibrated_preds[:, 0], average="binary", zero_division=0))
        uncalib_f1 = float(f1_score(y_true[:, 0], (y_prob[:, 0] >= 0.5).astype(int), average="binary", zero_division=0))
        opt_thresh = float(final_threshold_df.iloc[0]["optimal_threshold"])

        final_eval_summary = {
            "label": label,
            "pretty_name": pretty_name,
            "candidate_id": cfg["id"],
            "candidate_name": cfg["name"],
            "part": part_num,
            "best_epoch": best_epoch_num,
            "final_val_loss": float(final_val_res["loss"]),
            "final_val_macro_auprc": auprc,
            "final_val_macro_auroc": auroc,
            "final_val_macro_f1_calibrated": final_calibrated_f1,
            "final_val_micro_f1_calibrated": final_calibrated_micro_f1,
            "final_val_macro_f1_uncalibrated": uncalib_f1,
            "final_val_accuracy_calibrated": final_calibrated_acc,
            "final_val_balanced_acc_calibrated": final_balanced_acc,
            "optimal_threshold": opt_thresh,
        }

        write_json(run_dir / "best_val_eval_results.json", final_eval_summary)

        print("\n" + "=" * 65)
        print(f" RUN COMPLETED: {run_id.upper()} (PART {part_num})")
        print(f" Best Checkpoint (Epoch {best_epoch_num}):")
        print(f"   • Validation AUPRC:         {auprc:.4f}")
        print(f"   • Validation AUROC:         {auroc:.4f}")
        print(f"   • Validation Calibrated F1: {final_calibrated_f1:.4f}")
        print(f"   • Optimal Threshold:        {opt_thresh:.2f}")
        print(f"   • Validation Loss:          {final_eval_summary['final_val_loss']:.4f}")
        print("=" * 65 + "\n")

        if wandb_run is not None:
            wandb_run.summary["best_val/epoch"] = best_epoch_num
            wandb_run.summary["best_val/loss"] = final_eval_summary["final_val_loss"]
            wandb_run.summary["best_val/macro_auprc"] = auprc
            wandb_run.summary["best_val/macro_auroc"] = auroc
            wandb_run.summary["best_val/macro_f1_calibrated"] = final_calibrated_f1
            wandb_run.summary["best_val/optimal_threshold"] = opt_thresh
            wandb_run.summary["candidate/id"] = cfg["id"]
            wandb_run.summary["candidate/name"] = cfg["name"]
            wandb_run.summary["candidate/part"] = part_num
            wandb_run.finish()

        # Check and compute head-to-head comparison if competing candidate has completed
        check_and_save_comparison(label, output_dir)

        del model, optimizer, train_loader, val_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Final overall summary across all processed labels
    print_overall_summary(output_dir)
    print("\nFinal sweep execution batch finished successfully!")


if __name__ == "__main__":
    main()
