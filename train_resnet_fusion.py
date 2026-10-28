#!/usr/bin/env python3
"""
Baseline Training Runner: ResNet-50 + Clinical MLP Fusion.
Evaluates standard multimodal late fusion against the proposed Heterogeneous Graph Neural Network (GNN).

Using the exact same:
  - Vision Backbone: Pretrained ResNet-50 feature extractor
  - Tabular Clinical Inputs: Patient demographics (age, sex, eye, diabetes duration, insulin, top-20 comorbidities)
  - Loss: BCEWithLogitsLoss with positive class re-weighting (pos_weight) and log-prior bias initialization
  - Stratified Splits: image_train_12.csv, image_validation_12.csv, image_evaluation_12.csv
  - Evaluation Protocol: 150 epochs, AdamW (lr=1e-4), linear warmup, best validation checkpoint selection via Macro-AUPRC,
                         per-condition threshold calibration maximizing F1 (step=0.001), and held-out test evaluation.

Usage:
  # 1. Run Multilabel multimodal baseline (img_ctx, image_only, ctx_only) for Seed 42:
  python train_resnet_fusion.py --root data/BRSET --seed 42 --labels multilabel

  # 2. Run all conditions on cluster:
  sbatch submit_resnet_fusion.sh 42
"""

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

import wandb

from data_functions.BRSETFusionDataset import BRSETFusionDataset
from models.resnet_clinical_fusion import ResNetClinicalFusion
from utils.data_functions import calculate_positive_weights
from utils.feature_engineering import (
    preprocess_comorbidities,
    infer_demographic_columns,
    fit_demographic_specs,
)
from utils.metrics import find_optimal_thresholds, evaluate_multilabel_predictions
from utils.util_functions import select_device, set_seed, make_grad_scaler, autocast_context
from utils.logger import start_wandb_run


# =====================================================================
# 1. TARGET POLICIES & SCHEMAS
# =====================================================================

SINGLE_LABELS = [
    "increased_cup_disc",
    "drusens",
    "diabetic_retinopathy",
    "macular_edema",
    "scar",
    "amd",
    "myopic_fundus",
]

TARGET_POLICIES = {
    "myopic_fundus": {
        "tier": "local",
        "pretty_name": "Myopic Fundus",
        "hidden_dim": 128,
        "dropout": 0.25,
        "lr": 1e-4,
        "is_multilabel": False,
        "labels": ["myopic_fundus"],
    },
    "drusens": {
        "tier": "cluster",
        "pretty_name": "Drusen",
        "hidden_dim": 128,
        "dropout": 0.25,
        "lr": 1e-4,
        "is_multilabel": False,
        "labels": ["drusens"],
    },
    "diabetic_retinopathy": {
        "tier": "cluster",
        "pretty_name": "Diabetic Retinopathy",
        "hidden_dim": 512,
        "dropout": 0.15,
        "lr": 1e-4,
        "is_multilabel": False,
        "labels": ["diabetic_retinopathy"],
    },
    "increased_cup_disc": {
        "tier": "cluster",
        "pretty_name": "Glaucoma / Increased Cup-to-Disc",
        "hidden_dim": 512,
        "dropout": 0.15,
        "lr": 1e-4,
        "is_multilabel": False,
        "labels": ["increased_cup_disc"],
    },
    "macular_edema": {
        "tier": "cluster",
        "pretty_name": "Macular Edema",
        "hidden_dim": 512,
        "dropout": 0.15,
        "lr": 1e-4,
        "is_multilabel": False,
        "labels": ["macular_edema"],
    },
    "scar": {
        "tier": "cluster",
        "pretty_name": "Chorioretinal Scar",
        "hidden_dim": 512,
        "dropout": 0.15,
        "lr": 1e-4,
        "is_multilabel": False,
        "labels": ["scar"],
    },
    "amd": {
        "tier": "cluster",
        "pretty_name": "Age-Related Macular Degeneration (AMD)",
        "hidden_dim": 512,
        "dropout": 0.15,
        "lr": 1e-4,
        "is_multilabel": False,
        "labels": ["amd"],
    },
    "multilabel": {
        "tier": "cluster",
        "pretty_name": "Multilabel (All 7 Conditions)",
        "hidden_dim": 512,
        "dropout": 0.15,
        "lr": 1e-4,
        "is_multilabel": True,
        "labels": list(SINGLE_LABELS),
    },
}

ALL_TARGETS = list(TARGET_POLICIES.keys())
ALL_MODALITIES = ["ctx_only", "image_only", "img_ctx"]

TARGET_ALIASES = {
    "mf": "myopic_fundus",
    "myopic": "myopic_fundus",
    "myopic_fundus": "myopic_fundus",
    "drusen": "drusens",
    "drusens": "drusens",
    "dr": "diabetic_retinopathy",
    "diabetic_retinopathy": "diabetic_retinopathy",
    "icd": "increased_cup_disc",
    "glaucoma": "increased_cup_disc",
    "increased_cup_disc": "increased_cup_disc",
    "me": "macular_edema",
    "dme": "macular_edema",
    "macular_edema": "macular_edema",
    "scar": "scar",
    "amd": "amd",
    "multi": "multilabel",
    "multilabel": "multilabel",
    "mlc": "multilabel",
    "all_labels": "multilabel",
    "all": "multilabel",
}

MODALITY_ALIASES = {
    "ctx": "ctx_only",
    "context": "ctx_only",
    "ctx_only": "ctx_only",
    "context_only": "ctx_only",
    "img": "image_only",
    "image": "image_only",
    "image_only": "image_only",
    "img_only": "image_only",
    "img_ctx": "img_ctx",
    "image_ctx": "img_ctx",
    "multimodal": "img_ctx",
    "both": "img_ctx",
}


# =====================================================================
# 2. CLI ARGUMENTS
# =====================================================================

def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ResNet-50 + Clinical MLP Fusion Baseline Training Runner."
    )
    p.add_argument("--root", type=Path, required=True, help="Dataset root directory.")
    p.add_argument("--prepared-dir", type=Path, default=None, help="Directory containing train.csv/val.csv.")
    p.add_argument("--train-manifest", type=str, default="image_train_12.csv", help="Train manifest (default: image_train_12.csv).")
    p.add_argument("--validation-manifest", type=str, default="image_validation_12.csv", help="Validation manifest (default: image_validation_12.csv).")
    p.add_argument("--test-manifest", type=str, default="image_evaluation_12.csv", help="Held-out evaluation manifest (default: image_evaluation_12.csv).")
    p.add_argument("--skip-test", action="store_true", help="Skip evaluation on held-out test split.")
    p.add_argument("--output-dir", type=Path, default=None, help="Output root directory for model checkpoints.")

    # Model Type & Tier Filtering
    p.add_argument(
        "--tier",
        type=str,
        default="all",
        choices=["all", "local", "cluster"],
        help="Tier of models to run: 'local', 'cluster', or 'all' (default: all).",
    )
    p.add_argument(
        "--type",
        type=str,
        default="all",
        choices=["all", "singlelabel", "multilabel"],
        help="Filter model type: 'singlelabel', 'multilabel', or 'all' (default: all).",
    )
    p.add_argument(
        "--labels", "--label", "--targets",
        nargs="+",
        default=None,
        help="Filter specific target(s) (e.g., --labels multilabel, or --labels diabetic_retinopathy).",
    )
    p.add_argument(
        "--modalities", "--modality",
        nargs="+",
        default=None,
        help="Filter specific modality: 'ctx_only', 'image_only', 'img_ctx', or subset.",
    )

    # Architecture & Training Hyperparameters
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=150, help="Number of training epochs (default: 150).")
    p.add_argument("--batch-size", type=int, default=32, help="Effective batch size (default: 32).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5, help="Default fixed evaluation threshold (default: 0.5).")
    p.add_argument("--threshold-step", type=float, default=0.001, help="Sweep increment for optimal threshold calibration (default: 0.001).")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--force", action="store_true", help="Force retrain models even if completed checkpoints exist.")
    p.add_argument("--re-evaluate", action="store_true", help="Re-evaluate saved checkpoints without retraining.")

    # Splitting for SLURM Job Arrays
    p.add_argument("--split-total", type=int, default=1, help="Total number of parallel jobs.")
    p.add_argument("--split-index", type=int, default=0, help="0-indexed worker chunk to run.")

    # W&B Logging
    p.add_argument(
        "--wandb-project",
        type=str,
        default="ResNet-Fusion-Baseline",
        help="W&B project name for ResNet fusion baseline.",
    )
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])

    return p.parse_args(args)


# =====================================================================
# 3. RUN GENERATION & SUMMARY REPORTING
# =====================================================================

def build_planned_runs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    tier_choice = args.tier.lower()
    if tier_choice == "local":
        candidate_targets = [t for t, p in TARGET_POLICIES.items() if p["tier"] == "local"]
    elif tier_choice == "cluster":
        candidate_targets = [t for t, p in TARGET_POLICIES.items() if p["tier"] == "cluster"]
    else:
        candidate_targets = list(ALL_TARGETS)

    type_choice = args.type.lower()
    if type_choice == "singlelabel":
        candidate_targets = [t for t in candidate_targets if not TARGET_POLICIES[t]["is_multilabel"]]
    elif type_choice == "multilabel":
        candidate_targets = [t for t in candidate_targets if TARGET_POLICIES[t]["is_multilabel"]]

    if args.labels:
        requested_targets = []
        for raw in args.labels:
            clean = raw.lower().strip()
            canonical = TARGET_ALIASES.get(clean, clean)
            if canonical not in TARGET_POLICIES:
                raise ValueError(f"Unknown target '{raw}'. Available: {ALL_TARGETS}")
            if canonical not in requested_targets:
                requested_targets.append(canonical)
        target_keys = [t for t in requested_targets if t in candidate_targets]
        if not target_keys:
            raise ValueError(f"Requested targets {requested_targets} do not overlap with tier='{tier_choice}' / type='{type_choice}'.")
    else:
        target_keys = candidate_targets

    if args.modalities:
        target_modalities = []
        for raw in args.modalities:
            clean = raw.lower().strip()
            canonical = MODALITY_ALIASES.get(clean, clean)
            if canonical not in ALL_MODALITIES:
                raise ValueError(f"Unknown modality '{raw}'. Available: {ALL_MODALITIES}")
            if canonical not in target_modalities:
                target_modalities.append(canonical)
    else:
        target_modalities = list(ALL_MODALITIES)

    planned_runs = []
    run_idx = 1
    for target_key in target_keys:
        policy = TARGET_POLICIES[target_key]
        for modality in target_modalities:
            use_context = modality in ["ctx_only", "img_ctx"]
            use_image = modality in ["image_only", "img_ctx"]

            planned_runs.append({
                "target_key": target_key,
                "run_id": f"{target_key}_{modality}",
                "global_idx": run_idx,
                "pretty_name": policy["pretty_name"],
                "tier": policy["tier"],
                "is_multilabel": policy["is_multilabel"],
                "label_columns": policy["labels"],
                "modality": modality,
                "use_context": use_context,
                "use_image_features": use_image,
                "hidden_dim": policy["hidden_dim"],
                "dropout": policy["dropout"],
                "learning_rate": policy["lr"],
            })
            run_idx += 1

    return planned_runs


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def format_per_label_table(df_or_dict: Any, title: str) -> str:
    lines = [
        f"\n--- {title} ---",
        f"{'Label':<24} {'AUPRC':<9} {'AUROC':<9} {'F1':<9} {'Accuracy':<10} {'Bal_Acc':<10} {'Precision':<10} {'Recall':<9} {'Threshold':<9}",
        "-" * 105
    ]
    if isinstance(df_or_dict, pd.DataFrame):
        for _, r in df_or_dict.iterrows():
            lbl = r["label"]
            auprc_val = r.get("auprc", r.get("AUPRC", float("nan")))
            auroc_val = r.get("auroc", r.get("AUROC", float("nan")))
            f1_val = r.get("f1", r.get("F1", float("nan")))
            acc_val = r.get("accuracy", r.get("Accuracy", float("nan")))
            bal_acc_val = r.get("balanced_accuracy", r.get("Balanced_Accuracy", float("nan")))
            prec_val = r.get("precision", r.get("Precision", float("nan")))
            rec_val = r.get("recall", r.get("Recall", float("nan")))
            thresh_val = r.get("threshold", r.get("Optimal_Threshold", float("nan")))

            auprc_s = f"{float(auprc_val):.4f}" if not math.isnan(float(auprc_val)) else "N/A"
            auroc_s = f"{float(auroc_val):.4f}" if not math.isnan(float(auroc_val)) else "N/A"
            f1_s = f"{float(f1_val):.4f}" if not math.isnan(float(f1_val)) else "N/A"
            acc_s = f"{float(acc_val):.4f}" if not math.isnan(float(acc_val)) else "N/A"
            bal_acc_s = f"{float(bal_acc_val):.4f}" if not math.isnan(float(bal_acc_val)) else "N/A"
            prec_s = f"{float(prec_val):.4f}" if not math.isnan(float(prec_val)) else "N/A"
            rec_s = f"{float(rec_val):.4f}" if not math.isnan(float(rec_val)) else "N/A"
            thresh_s = f"{float(thresh_val):.4f}" if not math.isnan(float(thresh_val)) else "N/A"
            lines.append(f"{lbl:<24} {auprc_s:<9} {auroc_s:<9} {f1_s:<9} {acc_s:<10} {bal_acc_s:<10} {prec_s:<10} {rec_s:<9} {thresh_s:<9}")
    elif isinstance(df_or_dict, dict):
        for lbl, r in df_or_dict.items():
            auprc_val = r.get("AUPRC", r.get("auprc", float("nan")))
            auroc_val = r.get("AUROC", r.get("auroc", float("nan")))
            f1_val = r.get("F1", r.get("f1", r.get("Calibrated_F1", float("nan"))))
            acc_val = r.get("Accuracy", r.get("accuracy", float("nan")))
            bal_acc_val = r.get("Balanced_Accuracy", r.get("balanced_accuracy", float("nan")))
            prec_val = r.get("Precision", r.get("precision", float("nan")))
            rec_val = r.get("Recall", r.get("recall", float("nan")))
            thresh_val = r.get("Optimal_Threshold", r.get("threshold", float("nan")))

            auprc_s = f"{float(auprc_val):.4f}" if not math.isnan(float(auprc_val)) else "N/A"
            auroc_s = f"{float(auroc_val):.4f}" if not math.isnan(float(auroc_val)) else "N/A"
            f1_s = f"{float(f1_val):.4f}" if not math.isnan(float(f1_val)) else "N/A"
            acc_s = f"{float(acc_val):.4f}" if not math.isnan(float(acc_val)) else "N/A"
            bal_acc_s = f"{float(bal_acc_val):.4f}" if not math.isnan(float(bal_acc_val)) else "N/A"
            prec_s = f"{float(prec_val):.4f}" if not math.isnan(float(prec_val)) else "N/A"
            rec_s = f"{float(rec_val):.4f}" if not math.isnan(float(rec_val)) else "N/A"
            thresh_s = f"{float(thresh_val):.4f}" if not math.isnan(float(thresh_val)) else "N/A"
            lines.append(f"{lbl:<24} {auprc_s:<9} {auroc_s:<9} {f1_s:<9} {acc_s:<10} {bal_acc_s:<10} {prec_s:<10} {rec_s:<9} {thresh_s:<9}")
    lines.append("-" * 105)
    return "\n".join(lines)


def print_final_summary(output_dir: Path, seed: int) -> None:
    val_summary_data: Dict[str, Dict[str, Any]] = {}
    test_summary_data: Dict[str, Dict[str, Any]] = {}

    for target_key in ALL_TARGETS:
        target_dir = output_dir / target_key
        if not target_dir.exists():
            continue

        val_scores = {}
        test_scores = {}
        for mod in ALL_MODALITIES:
            v_path = target_dir / mod / "best_val_eval_results.json"
            if v_path.exists():
                try:
                    with open(v_path, "r") as f:
                        data = json.load(f)
                        val_scores[mod] = {
                            "macro_auprc": data.get("final_val_macro_auprc", 0.0),
                            "macro_auroc": data.get("final_val_macro_auroc", 0.0),
                            "calibrated_f1": data.get("final_val_macro_f1_calibrated", 0.0),
                            "accuracy": data.get("final_val_macro_accuracy", 0.0),
                            "balanced_accuracy": data.get("final_val_balanced_accuracy", 0.0),
                            "best_epoch": data.get("best_epoch", 0),
                            "per_label": data.get("per_label", data.get("per_label_metrics", {})),
                        }
                except Exception:
                    pass

            t_path = target_dir / mod / "test_evaluation_results.json"
            if t_path.exists():
                try:
                    with open(t_path, "r") as f:
                        tdata = json.load(f)
                        test_scores[mod] = {
                            "macro_auprc": tdata.get("test_macro_auprc", 0.0),
                            "macro_auroc": tdata.get("test_macro_auroc", 0.0),
                            "calibrated_f1": tdata.get("test_macro_f1_calibrated", 0.0),
                            "accuracy": tdata.get("test_macro_accuracy_calibrated", 0.0),
                            "balanced_accuracy": tdata.get("test_macro_balanced_accuracy", 0.0),
                            "best_epoch": tdata.get("best_epoch", 0),
                            "per_label": tdata.get("per_label", tdata.get("per_label_metrics", {})),
                        }
                except Exception:
                    pass

        if val_scores:
            val_summary_data[target_key] = val_scores
        if test_scores:
            test_summary_data[target_key] = test_scores

    if not val_summary_data and not test_summary_data:
        return

    # Table: Held-out Test Results (Macro AUPRC)
    if test_summary_data:
        print("\n" + "=" * 105)
        print(" RESNET-50 + CLINICAL MLP BASELINE: HELD-OUT TEST EVALUATION (MACRO AUPRC)")
        print("=" * 105)
        print(f"{'Target Model':<32} {'Tier':<8} {'Config':<12} {'ctx_only':<14} {'image_only':<14} {'img_ctx':<14} {'Best Modality':<14}")
        print("-" * 105)

        for target_key, mods in test_summary_data.items():
            pol = TARGET_POLICIES.get(target_key, {})
            pretty = pol.get("pretty_name", target_key)
            tier = pol.get("tier", "unknown")
            cfg_str = f"{pol.get('hidden_dim', 0)}h/{pol.get('dropout', 0.0)}d"

            ctx_s = f"{mods['ctx_only']['macro_auprc']:.4f}" if "ctx_only" in mods else "---"
            img_s = f"{mods['image_only']['macro_auprc']:.4f}" if "image_only" in mods else "---"
            both_s = f"{mods['img_ctx']['macro_auprc']:.4f}" if "img_ctx" in mods else "---"

            best_mod = "---"
            best_score = -1.0
            for m, vals in mods.items():
                if vals["macro_auprc"] > best_score:
                    best_score = vals["macro_auprc"]
                    best_mod = m

            print(f"{pretty:<32} {tier:<8} {cfg_str:<12} {ctx_s:<14} {img_s:<14} {both_s:<14} {best_mod:<14}")
        print("=" * 105)

        # Table: Calibrated F1
        print("\n" + "=" * 105)
        print(" RESNET-50 + CLINICAL MLP BASELINE: HELD-OUT TEST EVALUATION (CALIBRATED F1)")
        print("=" * 105)
        print(f"{'Target Model':<32} {'Tier':<8} {'Config':<12} {'ctx_only':<14} {'image_only':<14} {'img_ctx':<14} {'Best Modality':<14}")
        print("-" * 105)

        for target_key, mods in test_summary_data.items():
            pol = TARGET_POLICIES.get(target_key, {})
            pretty = pol.get("pretty_name", target_key)
            tier = pol.get("tier", "unknown")
            cfg_str = f"{pol.get('hidden_dim', 0)}h/{pol.get('dropout', 0.0)}d"

            ctx_s = f"{mods['ctx_only']['calibrated_f1']:.4f}" if "ctx_only" in mods else "---"
            img_s = f"{mods['image_only']['calibrated_f1']:.4f}" if "image_only" in mods else "---"
            both_s = f"{mods['img_ctx']['calibrated_f1']:.4f}" if "img_ctx" in mods else "---"

            best_mod = "---"
            best_score = -1.0
            for m, vals in mods.items():
                if vals["calibrated_f1"] > best_score:
                    best_score = vals["calibrated_f1"]
                    best_mod = m

            print(f"{pretty:<32} {tier:<8} {cfg_str:<12} {ctx_s:<14} {img_s:<14} {both_s:<14} {best_mod:<14}")
        print("=" * 105 + "\n")

    combined_summary = {
        "validation": val_summary_data,
        "test": test_summary_data,
    }
    write_json(output_dir / "final_models_summary.json", combined_summary)
    write_json(output_dir / f"seed_{seed}.json", combined_summary)


# =====================================================================
# 4. TRAINING & EVALUATION EPOCH RUNNER
# =====================================================================

def fusion_epoch(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    label_columns: List[str],
    modality: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Any = None,
    desc: str = "Processing",
    threshold: float = 0.5,
    accum_steps: int = 1,
) -> Dict[str, Any]:
    is_training = optimizer is not None
    model.train(is_training)

    use_image = modality in ["img_ctx", "image_only"]
    use_context = modality in ["img_ctx", "ctx_only"]

    total_loss = 0.0
    total_examples = 0
    all_targets = []
    all_probs = []

    pbar = tqdm(loader, desc=desc, leave=False)

    if is_training and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(pbar):
        imgs = batch["images"].to(device, non_blocking=True) if use_image else None
        cont = batch["continuous"].to(device, non_blocking=True) if use_context else None
        miss = batch["missing"].to(device, non_blocking=True) if use_context else None
        cat = batch["categorical"].to(device, non_blocking=True) if use_context else None
        targets = batch["targets"].to(device, non_blocking=True)

        with torch.set_grad_enabled(is_training):
            with autocast_context(amp_enabled):
                logits = model(images=imgs, continuous=cont, missing=miss, categorical=cat)
                loss = criterion(logits, targets)
                if is_training and accum_steps > 1:
                    loss_scaled = loss / accum_steps
                else:
                    loss_scaled = loss

            if is_training:
                if scaler is not None and amp_enabled:
                    scaler.scale(loss_scaled).backward()
                    if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                else:
                    loss_scaled.backward()
                    if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

        bs = int(targets.size(0))
        current_loss = float(loss.item())
        total_loss += current_loss * bs
        total_examples += bs

        pbar.set_postfix({"loss": f"{current_loss:.4f}"})

        all_targets.append(targets.detach().cpu().numpy())
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())

    np_targets = np.concatenate(all_targets, axis=0)
    np_probabilities = np.concatenate(all_probs, axis=0)

    if np_targets.ndim == 1 and np_probabilities.ndim > 1:
        np_targets = np_targets.reshape(-1, np_probabilities.shape[1])

    metrics_frame, summary, _ = evaluate_multilabel_predictions(
        targets=np_targets,
        probabilities=np_probabilities,
        label_names=label_columns,
        thresholds=threshold,
    )

    return {
        "loss": total_loss / max(total_examples, 1),
        "summary": summary,
        "targets": np_targets,
        "probabilities": np_probabilities,
        "metrics_frame": metrics_frame,
    }


# =====================================================================
# 5. MAIN ORCHESTRATOR
# =====================================================================

def main() -> None:
    args = parse_args()

    if args.split_index >= args.split_total:
        raise ValueError(f"--split-index ({args.split_index}) must be strictly less than --split-total ({args.split_total})")

    root = args.root.expanduser().resolve()
    prepared_dir = (args.prepared_dir or (root / "prepared")).expanduser().resolve()
    output_dir = (args.output_dir or (root / "runs" / f"resnet_fusion_{args.seed}")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed, args.deterministic)
    device = select_device(args.device)
    amp_enabled = not args.no_amp and device.type == "cuda"

    # 1. Build planned runs and slice across parallel workers
    all_runs = build_planned_runs(args)
    total_planned = len(all_runs)

    if total_planned <= args.split_total:
        if args.split_index < total_planned:
            my_runs = [all_runs[args.split_index]]
            start_idx = args.split_index
            end_idx = args.split_index + 1
        else:
            my_runs = []
            start_idx = total_planned
            end_idx = total_planned
    else:
        start_idx = total_planned * args.split_index // args.split_total
        end_idx = total_planned * (args.split_index + 1) // args.split_total
        my_runs = all_runs[start_idx:end_idx]

    print("===================================================================")
    print(" BASELINE TRAINING: RESNET-50 + CLINICAL MLP FUSION")
    print("===================================================================")
    print(f"Device: {device} | AMP: {amp_enabled} | Backbone: ResNet-50 (timm pretrained)")
    print(f"Dataset root: {root}")
    print(f"Output root directory: {output_dir}")
    print(f"W&B Project: {args.wandb_project}")
    print(f"Tier: {args.tier.upper()} | Type: {args.type.upper()} | Total Planned: {total_planned}")
    if len(my_runs) > 0:
        print(f"Worker Split {args.split_index + 1}/{args.split_total} - Processing models {start_idx + 1} to {end_idx} ({len(my_runs)} models).\n")
    else:
        print(f"Worker Split {args.split_index + 1}/{args.split_total} - No models allocated (0 models).\n")
        return

    # 2. Manifests & Clinical Preprocessing
    train_frame = pd.read_csv(prepared_dir / args.train_manifest)
    val_frame = pd.read_csv(prepared_dir / args.validation_manifest)

    train_frame, top_comorb = preprocess_comorbidities(train_frame, top_k=20)
    val_frame, _ = preprocess_comorbidities(val_frame, valid_comorb=top_comorb)

    demo_columns = infer_demographic_columns(train_frame, None, SINGLE_LABELS)
    demo_specs = fit_demographic_specs(train_frame, demo_columns)

    test_manifest_path = prepared_dir / args.test_manifest
    has_test_split = (not args.skip_test) and test_manifest_path.exists()
    if has_test_split:
        test_frame = pd.read_csv(test_manifest_path)
        test_frame, _ = preprocess_comorbidities(test_frame, valid_comorb=top_comorb)
        for col in demo_columns:
            if col not in test_frame.columns:
                test_frame[col] = 0.0
        print(f"Loaded Held-Out Test Set: '{args.test_manifest}' ({len(test_frame)} samples)\n")
    else:
        test_frame = None

    # Image Transforms (Standard ImageNet Normalization matching GNN)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
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

    # Dataset Cache
    current_cache_key = None
    train_dataset = None
    val_dataset = None
    test_dataset = None
    pos_weight = None
    adaptive_biases = None

    # Cardinality for categorical specs
    cat_specs = [s for s in demo_specs if s.kind == "categorical"]
    num_specs = [s for s in demo_specs if s.kind == "numeric"]
    cat_cards = [len(s.vocab) for s in cat_specs]

    # 3. Training Loop across allocated runs
    for run_meta in my_runs:
        target_key = run_meta["target_key"]
        pretty_name = run_meta["pretty_name"]
        tier = run_meta["tier"]
        modality = run_meta["modality"]
        run_id = run_meta["run_id"]
        global_idx = run_meta["global_idx"]
        is_multilabel = run_meta["is_multilabel"]
        label_columns = run_meta["label_columns"]
        num_labels = len(label_columns)

        h_dim = run_meta["hidden_dim"]
        dropout = run_meta["dropout"]
        learning_rate = run_meta["learning_rate"]
        use_context = run_meta["use_context"]
        use_image_features = run_meta["use_image_features"]

        load_images = use_image_features

        # Adaptive micro-batching: 512h uses micro_batch=16, accum=2 when images loaded
        if use_image_features and h_dim >= 512:
            micro_batch_size = max(1, args.batch_size // 2)
            accum_steps = max(1, args.batch_size // micro_batch_size)
        else:
            micro_batch_size = args.batch_size
            accum_steps = 1

        print("===================================================================")
        print(f"=== [BASELINE MODEL {global_idx}/{total_planned}] {target_key.upper()} | {modality.upper()} ===")
        print(f"=== Condition: {pretty_name} (Tier: {tier.upper()} | Multilabel: {is_multilabel}) ===")
        print(f"=== Config: fusion_dim={h_dim}, dropout={dropout}, lr={learning_rate} ===")
        print(f"=== Input: use_context={use_context}, use_image={use_image_features} (load_images={load_images}) ===")
        print(f"=== Micro-batch: {micro_batch_size}, Grad Accumulation: {accum_steps} ===")
        print("===================================================================\n")

        # Cache dataset per (target_tuple, load_images)
        cache_key = (tuple(label_columns), load_images)
        if cache_key != current_cache_key:
            del train_dataset, val_dataset, test_dataset
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            dataset_kwargs = {
                "image_dir": root / "fundus_photos_512",
                "target_cols": label_columns,
                "demo_cols": demo_columns,
                "demo_specs": demo_specs,
                "image_size": args.image_size,
                "load_images": load_images,
                "validate_paths": False,
            }

            train_dataset = BRSETFusionDataset(
                data=train_frame,
                transform=train_transform if load_images else None,
                **dataset_kwargs
            )
            val_dataset = BRSETFusionDataset(
                data=val_frame,
                transform=val_transform if load_images else None,
                **dataset_kwargs
            )
            if test_frame is not None:
                test_dataset = BRSETFusionDataset(
                    data=test_frame,
                    transform=val_transform if load_images else None,
                    **dataset_kwargs
                )
            else:
                test_dataset = None

            train_targets = train_dataset.target_tensor.numpy()
            pos_weight = calculate_positive_weights(train_targets).to(device)

            pos_counts = np.maximum(train_targets.sum(axis=0), 1.0)
            neg_counts = np.maximum(train_targets.shape[0] - pos_counts, 1.0)
            adaptive_biases = np.log(pos_counts / neg_counts).astype(np.float32)

            current_cache_key = cache_key

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=micro_batch_size,
            shuffle=True,
            num_workers=args.num_workers if load_images else 2,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=micro_batch_size,
            shuffle=False,
            num_workers=args.num_workers if load_images else 2,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=micro_batch_size,
            shuffle=False,
            num_workers=args.num_workers if load_images else 2,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        ) if test_dataset is not None else None

        run_dir = output_dir / target_key / modality
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "final_checkpoint.pt"
        best_auprc_path = run_dir / "checkpoint_best_macro_auprc.pt"

        if (run_dir / "test_evaluation_results.json").exists() and (run_dir / "best_val_eval_results.json").exists() and not args.force and not args.re_evaluate:
            print(f"[{run_id}] Already evaluated and complete on disk. Skipping.\n")
            continue

        # Initialize ResNet + Clinical MLP Fusion Model
        model = ResNetClinicalFusion(
            num_classes=num_labels,
            num_continuous=len(num_specs),
            cat_cardinalities=cat_cards,
            fusion_dim=h_dim,
            dropout=dropout,
            modality=modality,
            pretrained=True,
            initial_biases=adaptive_biases,
        ).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
        scaler = make_grad_scaler(amp_enabled)

        warmup_epochs = max(1, int(0.1 * args.epochs))
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )

        trial_config = {
            "baseline": "resnet50_clinical_mlp",
            "experiment": f"{target_key}_{modality}",
            "target": target_key,
            "label_pretty_name": pretty_name,
            "labels": label_columns,
            "tier": tier,
            "is_multilabel": is_multilabel,
            "modality": modality,
            "no_context": not use_context,
            "no_image": not use_image_features,
            "fusion_dim": h_dim,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": args.batch_size,
            "micro_batch_size": micro_batch_size,
            "accum_steps": accum_steps,
            "seed": args.seed,
            "epochs": args.epochs,
            "backbone": "resnet50",
            "image_size": args.image_size,
        }
        write_json(run_dir / "run_config.json", trial_config)

        group_name = "Baseline-Multilabel" if is_multilabel else f"Baseline-{target_key}"
        run_name = f"resnet_{target_key}_{modality}"
        tags = ["resnet_fusion_baseline", "multilabel" if is_multilabel else "singlelabel", target_key, modality]

        wandb_run = start_wandb_run(
            project=args.wandb_project,
            run_name=run_name,
            config=trial_config,
            output_dir=run_dir,
            entity=args.wandb_entity,
            group=group_name,
            tags=tags,
            mode=args.wandb_mode
        )

        best_metrics = {
            "macro_auprc": -1.0,
            "macro_auroc": -1.0,
            "macro_f1_calibrated": -1.0,
            "best_epoch": 1,
        }

        skip_training = best_auprc_path.exists() and not args.force
        num_train_epochs = 0 if skip_training else args.epochs
        if skip_training:
            print(f"--> Found existing checkpoint at {best_auprc_path}. Skipping training epochs...")

        # Training Epochs
        for epoch in range(1, num_train_epochs + 1):
            train_res = fusion_epoch(
                model=model, loader=train_loader, criterion=criterion, device=device,
                amp_enabled=amp_enabled, label_columns=label_columns, modality=modality,
                optimizer=optimizer, scaler=scaler, desc=f"[{run_id}] E{epoch:02d} Train",
                threshold=args.threshold, accum_steps=accum_steps
            )
            scheduler.step()

            val_res = fusion_epoch(
                model=model, loader=val_loader, criterion=criterion, device=device,
                amp_enabled=amp_enabled, label_columns=label_columns, modality=modality,
                optimizer=None, scaler=None, desc=f"[{run_id}] E{epoch:02d} Val",
                threshold=args.threshold
            )

            best_thresholds, epoch_threshold_df = find_optimal_thresholds(
                targets=val_res["targets"],
                probabilities=val_res["probabilities"],
                label_names=label_columns,
                step=args.threshold_step
            )
            epoch_calibrated_f1 = float(epoch_threshold_df["validation_f1"].mean())
            epoch_balanced_acc = float(epoch_threshold_df["validation_balanced_accuracy"].mean())

            current_macro_auprc = val_res["summary"]["macro_auprc"]
            current_macro_auroc = val_res["summary"]["macro_auroc"]
            is_best_auprc = current_macro_auprc > best_metrics["macro_auprc"]

            if is_best_auprc:
                best_metrics["macro_auprc"] = current_macro_auprc
                best_metrics["macro_auroc"] = current_macro_auroc
                best_metrics["macro_f1_calibrated"] = epoch_calibrated_f1
                best_metrics["best_epoch"] = epoch

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_res["loss"],
                    "macro_auprc": current_macro_auprc,
                    "macro_auroc": current_macro_auroc,
                    "calibrated_f1": epoch_calibrated_f1,
                    "balanced_accuracy": epoch_balanced_acc,
                    "optimal_thresholds": best_thresholds.tolist(),
                    "hyperparameters": trial_config,
                }, best_auprc_path)

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
                if is_multilabel and "metrics_frame" in val_res and val_res["metrics_frame"] is not None:
                    mf = val_res["metrics_frame"]
                    for _, row in mf.iterrows():
                        lbl_name = row["label"]
                        log_dict[f"val_label_auprc/{lbl_name}"] = float(row.get("auprc", row.get("AUPRC", 0.0)))
                        log_dict[f"val_label_auroc/{lbl_name}"] = float(row.get("auroc", row.get("AUROC", 0.0)))
                wandb.log(log_dict, step=epoch)

        if not skip_training:
            torch.save({
                "epoch": args.epochs,
                "model_state_dict": model.state_dict(),
                "best_metrics": best_metrics,
                "hyperparameters": trial_config,
            }, checkpoint_path)

        # Final Best Checkpoint Validation Evaluation
        if best_auprc_path.exists():
            ckpt = torch.load(best_auprc_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            best_epoch_num = ckpt.get("epoch", args.epochs)
        else:
            best_epoch_num = args.epochs

        final_val_res = fusion_epoch(
            model=model, loader=val_loader, criterion=criterion, device=device,
            amp_enabled=amp_enabled, label_columns=label_columns, modality=modality,
            optimizer=None, scaler=None, desc=f"[{run_id}] Final Val Eval", threshold=args.threshold
        )

        final_best_thresholds, final_threshold_df = find_optimal_thresholds(
            targets=final_val_res["targets"],
            probabilities=final_val_res["probabilities"],
            label_names=label_columns,
            step=args.threshold_step
        )

        val_calib_mf, val_calib_summary, _ = evaluate_multilabel_predictions(
            targets=final_val_res["targets"],
            probabilities=final_val_res["probabilities"],
            label_names=label_columns,
            thresholds=final_best_thresholds
        )

        final_calibrated_f1 = float(val_calib_summary["macro_f1"])
        final_balanced_acc = float(val_calib_summary["macro_balanced_accuracy"])

        final_results = {
            "target": target_key,
            "is_multilabel": is_multilabel,
            "labels": label_columns,
            "modality": modality,
            "tier": tier,
            "best_epoch": best_epoch_num,
            "final_val_loss": float(final_val_res["loss"]),
            "final_val_macro_auprc": float(val_calib_summary["macro_auprc"]),
            "final_val_macro_auroc": float(val_calib_summary["macro_auroc"]),
            "final_val_macro_f1_calibrated": final_calibrated_f1,
            "final_val_macro_f1_uncalibrated": float(final_val_res["summary"]["macro_f1"]),
            "final_val_macro_accuracy": float(val_calib_summary["macro_accuracy"]),
            "final_val_balanced_accuracy": final_balanced_acc,
            "optimal_thresholds": final_best_thresholds.tolist(),
            "config": trial_config,
            "per_label": {},
        }

        for _, r in val_calib_mf.iterrows():
            lbl = r["label"]
            final_results["per_label"][lbl] = {
                "AUPRC": float(r["auprc"]),
                "AUROC": float(r["auroc"]),
                "F1": float(r["f1"]),
                "Accuracy": float(r["accuracy"]),
                "Balanced_Accuracy": float(r["balanced_accuracy"]),
                "Precision": float(r["precision"]),
                "Recall": float(r["recall"]),
                "Optimal_Threshold": float(r["threshold"]),
                "TP": int(r["tp"]),
                "TN": int(r["tn"]),
                "FP": int(r["fp"]),
                "FN": int(r["fn"]),
            }
        final_results["per_label_metrics"] = final_results["per_label"]
        write_json(run_dir / "best_val_eval_results.json", final_results)

        print(f"\n[{run_id}] VALIDATION COMPLETE | Best Epoch: {best_epoch_num} | Val Macro AUPRC: {final_results['final_val_macro_auprc']:.4f} | Calibrated F1: {final_calibrated_f1:.4f}")
        print(format_per_label_table(val_calib_mf, f"[{run_id}] Validation Optimal Evaluation (Threshold Increment {args.threshold_step})"))

        # 4. Held-Out Test Evaluation using Validation-Calibrated Optimal Thresholds
        test_results = None
        if test_loader is not None:
            print(f"--> Running Held-Out Test Evaluation on '{args.test_manifest}'...")
            test_res = fusion_epoch(
                model=model, loader=test_loader, criterion=criterion, device=device,
                amp_enabled=amp_enabled, label_columns=label_columns, modality=modality,
                optimizer=None, scaler=None, desc=f"[{run_id}] Test Eval", threshold=args.threshold
            )

            test_calib_mf, test_calib_summary, _ = evaluate_multilabel_predictions(
                targets=test_res["targets"],
                probabilities=test_res["probabilities"],
                label_names=label_columns,
                thresholds=final_best_thresholds
            )

            _, test_uncalib_summary, _ = evaluate_multilabel_predictions(
                targets=test_res["targets"],
                probabilities=test_res["probabilities"],
                label_names=label_columns,
                thresholds=0.5
            )

            test_results = {
                "target": target_key,
                "is_multilabel": is_multilabel,
                "labels": label_columns,
                "modality": modality,
                "tier": tier,
                "best_epoch": best_epoch_num,
                "test_loss": float(test_res["loss"]),
                "test_macro_auprc": float(test_calib_summary["macro_auprc"]),
                "test_macro_auroc": float(test_calib_summary["macro_auroc"]),
                "test_macro_f1_calibrated": float(test_calib_summary["macro_f1"]),
                "test_micro_f1_calibrated": float(test_calib_summary["micro_f1"]),
                "test_macro_f1_uncalibrated": float(test_uncalib_summary["macro_f1"]),
                "test_macro_accuracy_calibrated": float(test_calib_summary["macro_accuracy"]),
                "test_macro_balanced_accuracy": float(test_calib_summary["macro_balanced_accuracy"]),
                "calibrated_thresholds": final_best_thresholds.tolist(),
                "per_label": {}
            }

            for _, r in test_calib_mf.iterrows():
                lbl = r["label"]
                test_results["per_label"][lbl] = {
                    "AUPRC": float(r["auprc"]),
                    "AUROC": float(r["auroc"]),
                    "F1": float(r["f1"]),
                    "Calibrated_F1": float(r["f1"]),
                    "Accuracy": float(r["accuracy"]),
                    "Balanced_Accuracy": float(r["balanced_accuracy"]),
                    "Precision": float(r["precision"]),
                    "Recall": float(r["recall"]),
                    "Optimal_Threshold": float(r["threshold"]),
                    "TP": int(r["tp"]),
                    "TN": int(r["tn"]),
                    "FP": int(r["fp"]),
                    "FN": int(r["fn"]),
                }
            test_results["per_label_metrics"] = test_results["per_label"]
            write_json(run_dir / "test_evaluation_results.json", test_results)
            print(f"[{run_id}] HELD-OUT TEST COMPLETE | Macro AUPRC: {test_results['test_macro_auprc']:.4f} | Calibrated F1: {test_results['test_macro_f1_calibrated']:.4f} | AUROC: {test_results['test_macro_auroc']:.4f}")
            print(format_per_label_table(test_calib_mf, f"[{run_id}] Held-Out Test Evaluation (Val-Calibrated Thresholds)"))

        if wandb_run is not None:
            wandb.run.summary["best_val_macro_auprc"] = final_results["final_val_macro_auprc"]
            wandb.run.summary["best_val_macro_auroc"] = final_results["final_val_macro_auroc"]
            wandb.run.summary["best_val_macro_f1_calibrated"] = final_calibrated_f1
            wandb.run.summary["best_epoch"] = best_epoch_num

            if test_results is not None:
                wandb.run.summary["test/loss"] = test_results["test_loss"]
                wandb.run.summary["test/macro_auprc"] = test_results["test_macro_auprc"]
                wandb.run.summary["test/macro_auroc"] = test_results["test_macro_auroc"]
                wandb.run.summary["test/macro_f1_calibrated"] = test_results["test_macro_f1_calibrated"]
                wandb.run.summary["test/macro_accuracy"] = test_results["test_macro_accuracy_calibrated"]
                wandb.run.summary["test/macro_balanced_accuracy"] = test_results["test_macro_balanced_accuracy"]

            wandb.finish()

        del model, optimizer, scheduler, criterion, train_loader, val_loader
        if test_loader is not None:
            del test_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Print summary and write final_models_summary.json / seed_{seed}.json
    print_final_summary(output_dir, args.seed)


if __name__ == "__main__":
    main()
