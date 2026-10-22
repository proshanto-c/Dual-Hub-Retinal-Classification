#!/usr/bin/env python3
"""
Final Model Training Runner for Single-Label and Multilabel Fundus Classification.

Hyperparameter Policy:
  - 128 hidden, 0.25 dropout (Tier: Local):
      • myopic_fundus
      • drusens
  - 512 hidden, 0.15 dropout (Tier: Cluster):
      • diabetic_retinopathy
      • increased_cup_disc
      • macular_edema
      • scar
      • amd
      • multilabel (all 7 conditions simultaneously)

Modalities per target:
  1. ctx_only   (use_context=True,  use_image_features=False)
  2. image_only (use_context=False, use_image_features=True)
  3. img_ctx    (use_context=True,  use_image_features=True)

Totals:
  - Local Tier:    2 labels x 3 modalities = 6 models
  - Cluster Tier:  6 targets (5 singlelabel + 1 multilabel) x 3 modalities = 18 models
  - Study Total:   24 models

Usage Examples:
  # Run all 6 local models (myopic_fundus, drusens across 3 modalities):
  python train_final_models.py --root "data/BRSET" --tier local

  # Run only multilabel models (ctx_only, image_only, img_ctx):
  python train_final_models.py --root "data/BRSET" --labels multilabel

  # Run all 18 cluster models via SLURM:
  sbatch submit_final_models.sh 42
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms

import wandb

from data_functions.BRSETGraphDataset import BRSETGraphDataset
from gnn.patientgraphmodel import PatientGraphModel
from utils.data_functions import create_dataloader, calculate_positive_weights
from utils.feature_engineering import (
    preprocess_comorbidities,
    infer_demographic_columns,
    fit_demographic_specs,
)
from utils.train_functions import graph_epoch
from utils.metrics import find_optimal_thresholds, evaluate_multilabel_predictions
from utils.util_functions import select_device, set_seed, make_grad_scaler
from utils.logger import start_wandb_run


# =====================================================================
# 1. FINAL TARGET POLICIES & SCHEMAS
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
    # --- Tier: LOCAL (128h / 0.25d) ---
    "myopic_fundus": {
        "tier": "local",
        "pretty_name": "Myopic Fundus",
        "hidden_dim": 128,
        "dropout": 0.25,
        "lr": 1e-4,
        "num_layers": 3,
        "heads": 2,
        "is_multilabel": False,
        "labels": ["myopic_fundus"],
    },
    "drusens": {
        "tier": "local",
        "pretty_name": "Drusen",
        "hidden_dim": 128,
        "dropout": 0.25,
        "lr": 1e-4,
        "num_layers": 3,
        "heads": 2,
        "is_multilabel": False,
        "labels": ["drusens"],
    },

    # --- Tier: CLUSTER (512h / 0.15d) Single-Label ---
    "diabetic_retinopathy": {
        "tier": "cluster",
        "pretty_name": "Diabetic Retinopathy",
        "hidden_dim": 512,
        "dropout": 0.15,
        "lr": 1e-4,
        "num_layers": 3,
        "heads": 2,
        "is_multilabel": False,
        "labels": ["diabetic_retinopathy"],
    },
    "increased_cup_disc": {
        "tier": "cluster",
        "pretty_name": "Glaucoma / Increased Cup-to-Disc",
        "hidden_dim": 512,
        "dropout": 0.15,
        "lr": 1e-4,
        "num_layers": 3,
        "heads": 2,
        "is_multilabel": False,
        "labels": ["increased_cup_disc"],
    },
    "macular_edema": {
        "tier": "cluster",
        "pretty_name": "Macular Edema",
        "hidden_dim": 512,
        "dropout": 0.15,
        "lr": 1e-4,
        "num_layers": 3,
        "heads": 2,
        "is_multilabel": False,
        "labels": ["macular_edema"],
    },
    "scar": {
        "tier": "cluster",
        "pretty_name": "Chorioretinal Scar",
        "hidden_dim": 512,
        "dropout": 0.15,
        "lr": 1e-4,
        "num_layers": 3,
        "heads": 2,
        "is_multilabel": False,
        "labels": ["scar"],
    },
    "amd": {
        "tier": "cluster",
        "pretty_name": "Age-Related Macular Degeneration (AMD)",
        "hidden_dim": 512,
        "dropout": 0.15,
        "lr": 1e-4,
        "num_layers": 3,
        "heads": 2,
        "is_multilabel": False,
        "labels": ["amd"],
    },

    # --- Tier: CLUSTER (512h / 0.15d) Multilabel ---
    "multilabel": {
        "tier": "cluster",
        "pretty_name": "Multilabel (All 7 Conditions)",
        "hidden_dim": 512,
        "dropout": 0.15,
        "lr": 1e-4,
        "num_layers": 3,
        "heads": 2,
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
        description="Final training runner for single-label and multilabel fundus models across ctx_only, image_only, and img_ctx."
    )
    p.add_argument("--root", type=Path, required=True, help="Dataset root directory.")
    p.add_argument("--prepared-dir", type=Path, default=None, help="Directory containing train.csv/val.csv.")
    p.add_argument("--train-manifest", type=str, default=None, help="Train manifest (defaults to image_train_<seed>.csv or image_train_12.csv).")
    p.add_argument("--validation-manifest", type=str, default=None, help="Validation manifest (defaults to image_validation_<seed>.csv or image_validation_12.csv).")
    p.add_argument("--test-manifest", type=str, default=None, help="Held-out evaluation manifest (defaults to image_evaluation_<seed>.csv or image_evaluation_12.csv).")
    p.add_argument("--skip-test", action="store_true", help="Skip evaluation on the held-out test split.")
    p.add_argument("--output-dir", type=Path, default=None, help="Output root directory for model checkpoints.")

    # Model Type & Tier Filtering
    p.add_argument(
        "--tier",
        type=str,
        default="all",
        choices=["all", "local", "cluster"],
        help="Tier of models to run: 'local' (128h: MF, Drusen), 'cluster' (512h: DR, ICD, ME, Scar, AMD, Multilabel), or 'all' (default: all).",
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
        help="Filter specific target(s) (e.g., --labels myopic_fundus drusens, or --labels multilabel).",
    )
    p.add_argument(
        "--modalities", "--modality",
        nargs="+",
        default=None,
        help="Filter specific modality/modalities: 'ctx_only', 'image_only', 'img_ctx', or any subset.",
    )

    # Architecture & Training Hyperparameters
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
    p.add_argument("--force", action="store_true", help="Force retrain models even if completed checkpoints exist.")

    # Splitting for SLURM Job Arrays
    p.add_argument("--split-total", type=int, default=1, help="Total number of parallel jobs.")
    p.add_argument("--split-index", type=int, default=0, help="0-indexed worker chunk to run.")

    # W&B Logging
    p.add_argument(
        "--wandb-project",
        type=str,
        default="Final-Fundus-Models",
        help="W&B project name (all models log to this unified project).",
    )
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])

    return p.parse_args(args)


# =====================================================================
# 3. RUN GENERATION & SUMMARY REPORTING
# =====================================================================

def build_planned_runs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Constructs the list of models to train based on tier, type, label, and modality filters."""
    # 1. Filter targets by tier
    tier_choice = args.tier.lower()
    if tier_choice == "local":
        candidate_targets = [t for t, p in TARGET_POLICIES.items() if p["tier"] == "local"]
    elif tier_choice == "cluster":
        candidate_targets = [t for t, p in TARGET_POLICIES.items() if p["tier"] == "cluster"]
    else:
        candidate_targets = list(ALL_TARGETS)

    # 2. Filter targets by type (singlelabel vs multilabel)
    type_choice = args.type.lower()
    if type_choice == "singlelabel":
        candidate_targets = [t for t in candidate_targets if not TARGET_POLICIES[t]["is_multilabel"]]
    elif type_choice == "multilabel":
        candidate_targets = [t for t in candidate_targets if TARGET_POLICIES[t]["is_multilabel"]]

    # 3. Filter targets by explicit CLI list if provided
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
            raise ValueError(f"Requested targets {requested_targets} have no overlap with selected tier '{tier_choice}' / type '{type_choice}'.")
    else:
        target_keys = candidate_targets

    # 4. Filter modalities
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

    # 5. Construct run metadata list
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
                "num_layers": policy["num_layers"],
                "heads": policy["heads"],
            })
            run_idx += 1

    return planned_runs


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def print_final_summary(output_dir: Path) -> None:
    """Scans output_dir for completed model evaluation results and prints consolidated validation and test comparison tables."""
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
                            "best_epoch": data.get("best_epoch", 0),
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
                            "best_epoch": tdata.get("best_epoch", 0),
                        }
                except Exception:
                    pass

        if val_scores:
            val_summary_data[target_key] = val_scores
        if test_scores:
            test_summary_data[target_key] = test_scores

    if not val_summary_data and not test_summary_data:
        return

    # Table 1: Validation Results
    if val_summary_data:
        print("\n" + "=" * 102)
        print(" FINAL TRAINING: VALIDATION PERFORMANCE (MACRO AUPRC COMPARISON)")
        print("=" * 102)
        print(f"{'Target Model':<32} {'Tier':<8} {'Config':<12} {'ctx_only':<14} {'image_only':<14} {'img_ctx':<14} {'Best Modality':<14}")
        print("-" * 102)

        for target_key, mods in val_summary_data.items():
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
        print("=" * 102)

    # Table 2: Held-out Test Results
    if test_summary_data:
        print("\n" + "=" * 102)
        print(" FINAL TRAINING: HELD-OUT TEST EVALUATION (MACRO AUPRC COMPARISON)")
        print("=" * 102)
        print(f"{'Target Model':<32} {'Tier':<8} {'Config':<12} {'ctx_only':<14} {'image_only':<14} {'img_ctx':<14} {'Best Modality':<14}")
        print("-" * 102)

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
        print("=" * 102 + "\n")

    combined_summary = {
        "validation": val_summary_data,
        "test": test_summary_data,
    }
    write_json(output_dir / "final_models_summary.json", combined_summary)


# =====================================================================
# 4. MAIN ORCHESTRATOR
# =====================================================================

def main() -> None:
    args = parse_args()

    if args.patch_size is None:
        args.patch_size = 16 if args.backbone == "retfound" else 32

    if args.split_index >= args.split_total:
        raise ValueError(f"--split-index ({args.split_index}) must be strictly less than --split-total ({args.split_total})")

    root = args.root.expanduser().resolve()
    prepared_dir = (args.prepared_dir or (root / "prepared")).expanduser().resolve()
    output_dir = (args.output_dir or (root / "runs" / f"final_models_{args.seed}")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed, args.deterministic)
    device = select_device(args.device)
    amp_enabled = not args.no_amp and device.type == "cuda"

    # 1. Build planned runs and perform guaranteed non-empty worker slicing
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
    print(" FINAL MODEL TRAINING (SINGLE-LABEL & MULTILABEL)")
    print("===================================================================")
    print(f"Device: {device} | AMP: {amp_enabled} | Backbone: {args.backbone}")
    print(f"Dataset root: {root}")
    print(f"Output root directory: {output_dir}")
    print(f"W&B Unified Project: {args.wandb_project}")
    print(f"Tier: {args.tier.upper()} | Type: {args.type.upper()} | Total Planned: {total_planned}")
    if len(my_runs) > 0:
        print(f"Worker Split {args.split_index + 1}/{args.split_total} - Processing models {start_idx + 1} to {end_idx} ({len(my_runs)} models).\n")
    else:
        print(f"Worker Split {args.split_index + 1}/{args.split_total} - No models allocated (0 models).\n")

    if len(my_runs) == 0:
        print("No models allocated for this worker split. Exiting.")
        return

    # 2. Resolve Manifests, Preprocess Clinical Context, and Load Test Split
    if args.train_manifest is None:
        cand = prepared_dir / f"image_train_{args.seed}.csv"
        args.train_manifest = cand.name if cand.exists() else "image_train_12.csv"
    if args.validation_manifest is None:
        cand = prepared_dir / f"image_validation_{args.seed}.csv"
        args.validation_manifest = cand.name if cand.exists() else "image_validation_12.csv"
    if args.test_manifest is None:
        cand = prepared_dir / f"image_evaluation_{args.seed}.csv"
        args.test_manifest = cand.name if cand.exists() else "image_evaluation_12.csv"

    print(f"Manifests: Train='{args.train_manifest}' | Val='{args.validation_manifest}' | Test='{args.test_manifest}'")

    train_frame = pd.read_csv(prepared_dir / args.train_manifest)
    val_frame = pd.read_csv(prepared_dir / args.validation_manifest)

    train_frame, top_comorb = preprocess_comorbidities(train_frame, top_k=20)
    val_frame, _ = preprocess_comorbidities(val_frame, valid_comorb=top_comorb)

    # Demographic columns & specs fitted strictly on training data
    demo_columns = infer_demographic_columns(train_frame, None, SINGLE_LABELS)
    demo_specs = fit_demographic_specs(train_frame, demo_columns)

    # Load Held-Out Test Set (unless --skip-test is explicitly passed)
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
        if not args.skip_test:
            print(f"Warning: Held-out test manifest not found at {test_manifest_path}. Skipping test evaluation.\n")

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

    # Cache dataset per (target_tuple, load_images) combo
    current_cache_key = None
    train_dataset = None
    val_dataset = None
    test_dataset = None
    pos_weight = None
    adaptive_biases = None

    # 3. Training Loop
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
        n_layers = run_meta["num_layers"]
        heads = run_meta["heads"]
        use_context = run_meta["use_context"]
        use_image_features = run_meta["use_image_features"]

        # Optimize: Avoid reading raw images from disk when training ctx_only
        load_images = use_image_features

        # Adaptive micro-batching: 512h uses micro_batch=16, accum=2 when images are loaded
        if use_image_features and h_dim >= 512:
            micro_batch_size = max(1, args.batch_size // 2)
            accum_steps = max(1, args.batch_size // micro_batch_size)
        else:
            micro_batch_size = args.batch_size
            accum_steps = 1

        print("===================================================================")
        print(f"=== [MODEL {global_idx}/{total_planned}] {target_key.upper()} | {modality.upper()} ===")
        print(f"=== Condition: {pretty_name} (Tier: {tier.upper()} | Multilabel: {is_multilabel}) ===")
        print(f"=== Config: h_dim={h_dim}, dropout={dropout}, lr={learning_rate} (uniform) ===")
        print(f"=== Input: use_context={use_context}, use_image_features={use_image_features} (load_images={load_images}) ===")
        print(f"=== Micro-batch: {micro_batch_size}, Grad Accumulation: {accum_steps} ===")
        print("===================================================================\n")

        # Invalidate dataset cache if target labels or load_images changed
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
                "patch_size": args.patch_size,
                "connectivity": args.connectivity,
                "prediction_level": "dual_hub",
                "train_frame": train_frame,
                "load_images": load_images,
            }

            train_dataset = BRSETGraphDataset(
                data=train_frame,
                validate_paths=False,
                transform=train_transform if load_images else None,
                **dataset_kwargs
            )

            val_dataset = BRSETGraphDataset(
                data=val_frame,
                validate_paths=False,
                transform=val_transform if load_images else None,
                **dataset_kwargs
            )

            if test_frame is not None:
                test_dataset = BRSETGraphDataset(
                    data=test_frame,
                    validate_paths=False,
                    transform=val_transform if load_images else None,
                    **dataset_kwargs
                )
            else:
                test_dataset = None

            train_targets = np.vstack([s["label_vector"] for s in train_dataset.samples])
            pos_weight = calculate_positive_weights(train_targets).to(device)

            pos_counts = np.maximum(train_targets.sum(axis=0), 1.0)
            neg_counts = np.maximum(train_targets.shape[0] - pos_counts, 1.0)
            adaptive_biases = np.log(pos_counts / neg_counts).astype(np.float32)

            current_cache_key = cache_key

            total_samples = len(train_dataset.samples)
            if is_multilabel:
                print(f"Target Statistics for Multilabel ({num_labels} labels, {total_samples} samples):")
                for l_idx, lbl in enumerate(label_columns):
                    p_c = int(pos_counts[l_idx])
                    pct = (p_c / total_samples) * 100.0
                    print(f"  • {lbl:<24}: {p_c:4d}/{total_samples} ({pct:5.2f}%) | Pos Wt: {pos_weight[l_idx].item():.2f} | Prior Bias: {adaptive_biases[l_idx]:.2f}")
                print()
            else:
                p_count = int(pos_counts[0])
                pct = (p_count / total_samples) * 100.0
                print(f"Target Statistics for [{target_key}]: {p_count}/{total_samples} positives ({pct:.2f}%) | Pos Weight: {pos_weight[0].item():.2f} | Prior Bias: {adaptive_biases[0]:.2f}\n")

        # Dataloaders
        train_loader = create_dataloader(
            dataset=train_dataset, batch_size=micro_batch_size, shuffle=True,
            num_workers=args.num_workers if load_images else 2,
            pin_memory=(device.type == "cuda"),
            seed=args.seed + global_idx, is_graph=True
        )
        val_loader = create_dataloader(
            dataset=val_dataset, batch_size=micro_batch_size, shuffle=False,
            num_workers=args.num_workers if load_images else 2,
            pin_memory=(device.type == "cuda"),
            seed=args.seed, is_graph=True
        )
        if test_dataset is not None:
            test_loader = create_dataloader(
                dataset=test_dataset, batch_size=micro_batch_size, shuffle=False,
                num_workers=args.num_workers if load_images else 2,
                pin_memory=(device.type == "cuda"),
                seed=args.seed, is_graph=True
            )
        else:
            test_loader = None

        run_dir = output_dir / target_key / modality
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "final_checkpoint.pt"
        best_auprc_path = run_dir / "checkpoint_best_macro_auprc.pt"

        # Check if already fully evaluated and complete
        if (run_dir / "test_evaluation_results.json").exists() and (run_dir / "best_val_eval_results.json").exists() and not args.force:
            print(f"[{run_id}] Already evaluated and complete on disk. Skipping (use --force to retrain).\n")
            continue

        # Initialize Heterogeneous Model
        model = PatientGraphModel(
            num_labels=num_labels,
            demographic_specs=demo_specs,
            hidden_dim=h_dim,
            num_layers=n_layers,
            dropout=dropout,
            heads=heads,
            prediction_level="dual_hub",
            initial_biases=adaptive_biases,
            use_context=use_context,
            use_image_features=use_image_features,
            backbone=args.backbone,
            patch_connectivity=args.connectivity,
            image_size=args.image_size,
            patch_size=args.patch_size,
        ).to(device)

        if getattr(args, "compile", False) and os.name != "nt":
            try:
                model = torch.compile(model)
            except Exception as e:
                print(f"Skipping torch.compile: {e}")

        raw_model = getattr(model, "_orig_mod", model)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
        scaler = make_grad_scaler(amp_enabled)

        warmup_epochs = max(1, int(0.1 * args.epochs))
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )

        # Standardized schema compatible with evaluate_all.py
        trial_config = {
            "experiment": f"{target_key}_{modality}",
            "target": target_key,
            "label_pretty_name": pretty_name,
            "labels": label_columns,
            "tier": tier,
            "is_multilabel": is_multilabel,
            "modality": modality,
            "no_context": not use_context,
            "no_image": not use_image_features,
            "h_dim": h_dim,
            "hidden_dim": h_dim,
            "num_layers": n_layers,
            "n_layers": n_layers,
            "heads": heads,
            "dropout": dropout,
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
            "demo_specs": [ds.to_dict() if hasattr(ds, "to_dict") else ds.__dict__ for ds in demo_specs],
            "demo_columns": demo_columns,
        }
        write_json(run_dir / "run_config.json", trial_config)

        # W&B: Unified project, grouped by target
        group_name = "Multilabel" if is_multilabel else f"Singlelabel-{target_key}"
        run_name = f"{target_key}_{modality}"
        tags = ["final_training", "multilabel" if is_multilabel else "singlelabel", target_key, modality, tier]

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
            "min_val_loss": float("inf")
        }

        # Check if model has a saved checkpoint to skip training and jump to evaluation
        skip_training = best_auprc_path.exists() and not args.force
        num_train_epochs = 0 if skip_training else args.epochs
        if skip_training:
            print(f"--> Found existing checkpoint at {best_auprc_path}. Skipping training epochs and proceeding directly to final evaluation...")

        # Training Epochs
        for epoch in range(1, num_train_epochs + 1):
            train_res = graph_epoch(
                model=model, loader=train_loader, criterion=criterion, device=device,
                amp_enabled=amp_enabled, label_columns=label_columns, optimizer=optimizer,
                scaler=scaler, desc=f"[{run_id}] E{epoch:02d} Train", threshold=args.threshold,
                accum_steps=accum_steps
            )
            scheduler.step()

            val_res = graph_epoch(
                model=model, loader=val_loader, criterion=criterion, device=device,
                amp_enabled=amp_enabled, label_columns=label_columns, optimizer=None,
                scaler=None, desc=f"[{run_id}] E{epoch:02d} Val", threshold=args.threshold
            )

            best_thresholds, epoch_threshold_df = find_optimal_thresholds(
                targets=val_res["targets"],
                probabilities=val_res["probabilities"],
                label_names=label_columns,
                step=0.05
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

                # Save best checkpoint
                raw_model = getattr(model, "_orig_mod", model)
                best_ckpt_path = run_dir / "checkpoint_best_macro_auprc.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "val_loss": val_res["loss"],
                    "macro_auprc": current_macro_auprc,
                    "macro_auroc": current_macro_auroc,
                    "calibrated_f1": epoch_calibrated_f1,
                    "balanced_accuracy": epoch_balanced_acc,
                    "optimal_thresholds": best_thresholds.tolist(),
                    "hyperparameters": trial_config,
                    "hyperparams": trial_config,  # Compatibility with evaluate_all.py
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
                # Log individual label AUPRCs for multilabel models
                if is_multilabel and "metrics_frame" in val_res and val_res["metrics_frame"] is not None:
                    mf = val_res["metrics_frame"]
                    for _, row in mf.iterrows():
                        lbl_name = row["label"]
                        log_dict[f"val_label_auprc/{lbl_name}"] = float(row.get("auprc", row.get("AUPRC", 0.0)))
                        log_dict[f"val_label_auroc/{lbl_name}"] = float(row.get("auroc", row.get("AUROC", 0.0)))

                wandb.log(log_dict, step=epoch)

        # Save final checkpoint if training was performed
        if not skip_training:
            raw_model = getattr(model, "_orig_mod", model)
            torch.save({
                "epoch": args.epochs,
                "model_state_dict": raw_model.state_dict(),
                "best_metrics": best_metrics,
                "hyperparameters": trial_config,
                "hyperparams": trial_config,
            }, checkpoint_path)

        # Final evaluation on the best AUPRC checkpoint
        best_auprc_path = run_dir / "checkpoint_best_macro_auprc.pt"
        if best_auprc_path.exists():
            ckpt = torch.load(best_auprc_path, map_location=device, weights_only=False)
            raw_model.load_state_dict(ckpt["model_state_dict"], strict=False)
            best_epoch_num = ckpt.get("epoch", args.epochs)
        else:
            best_epoch_num = args.epochs

        final_val_res = graph_epoch(
            model=model, loader=val_loader, criterion=criterion, device=device,
            amp_enabled=amp_enabled, label_columns=label_columns, optimizer=None,
            scaler=None, desc=f"[{run_id}] Final Best Val Eval", threshold=args.threshold
        )

        final_best_thresholds, final_threshold_df = find_optimal_thresholds(
            targets=final_val_res["targets"],
            probabilities=final_val_res["probabilities"],
            label_names=label_columns,
            step=0.05
        )

        final_calibrated_f1 = float(final_threshold_df["validation_f1"].mean())
        final_balanced_acc = float(final_threshold_df["validation_balanced_accuracy"].mean())

        final_results = {
            "target": target_key,
            "is_multilabel": is_multilabel,
            "labels": label_columns,
            "modality": modality,
            "tier": tier,
            "best_epoch": best_epoch_num,
            "final_val_loss": final_val_res["loss"],
            "final_val_macro_auprc": final_val_res["summary"]["macro_auprc"],
            "final_val_macro_auroc": final_val_res["summary"]["macro_auroc"],
            "final_val_macro_f1_calibrated": final_calibrated_f1,
            "final_val_macro_f1_uncalibrated": final_val_res["summary"]["macro_f1"],
            "final_val_balanced_accuracy": final_balanced_acc,
            "optimal_thresholds": final_best_thresholds.tolist(),
            "config": trial_config,
        }

        # Per-label detailed metrics
        if "metrics_frame" in final_val_res and final_val_res["metrics_frame"] is not None:
            per_label_dict = {}
            for _, r in final_val_res["metrics_frame"].iterrows():
                lbl = r["label"]
                per_label_dict[lbl] = {
                    "AUPRC": float(r.get("auprc", r.get("AUPRC", 0.0))),
                    "AUROC": float(r.get("auroc", r.get("AUROC", 0.0))),
                    "F1": float(r.get("f1", r.get("F1", 0.0))),
                }
            final_results["per_label_metrics"] = per_label_dict

        write_json(run_dir / "best_val_eval_results.json", final_results)

        print(f"\n[{run_id}] COMPLETE | Best Epoch: {best_epoch_num} | Val Macro AUPRC: {final_results['final_val_macro_auprc']:.4f} | Calibrated F1: {final_calibrated_f1:.4f}\n")

        # 4. Evaluation on Held-Out Test Set using Validation-Calibrated Optimal Thresholds
        test_results = None
        if test_loader is not None:
            print(f"--> Running Held-Out Test Evaluation on '{args.test_manifest}'...")
            test_res = graph_epoch(
                model=model, loader=test_loader, criterion=criterion, device=device,
                amp_enabled=amp_enabled, label_columns=label_columns, optimizer=None,
                scaler=None, desc=f"[{run_id}] Test Eval", threshold=args.threshold
            )

            # Evaluate with validation-calibrated optimal thresholds
            test_calib_mf, test_calib_summary, _ = evaluate_multilabel_predictions(
                targets=test_res["targets"],
                probabilities=test_res["probabilities"],
                label_names=label_columns,
                thresholds=final_best_thresholds
            )

            # Evaluate with default uncalibrated 0.5 threshold
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
                test_results["per_label"][r["label"]] = {
                    "AUPRC": float(r["auprc"]),
                    "AUROC": float(r["auroc"]),
                    "Calibrated_F1": float(r["f1"]),
                    "Precision": float(r["precision"]),
                    "Recall": float(r["recall"]),
                    "Accuracy": float(r["accuracy"]),
                    "Balanced_Accuracy": float(r["balanced_accuracy"]),
                    "Optimal_Threshold": float(r["threshold"]),
                    "TP": int(r["tp"]),
                    "TN": int(r["tn"]),
                    "FP": int(r["fp"]),
                    "FN": int(r["fn"]),
                }

            write_json(run_dir / "test_evaluation_results.json", test_results)
            print(f"[{run_id}] HELD-OUT TEST RESULTS | Macro AUPRC: {test_results['test_macro_auprc']:.4f} | Calibrated F1: {test_results['test_macro_f1_calibrated']:.4f} | AUROC: {test_results['test_macro_auroc']:.4f}\n")

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
                wandb.run.summary["test/micro_f1_calibrated"] = test_results["test_micro_f1_calibrated"]
                wandb.run.summary["test/macro_f1_uncalibrated"] = test_results["test_macro_f1_uncalibrated"]
                wandb.run.summary["test/macro_balanced_accuracy"] = test_results["test_macro_balanced_accuracy"]

                for lbl_name, l_metrics in test_results["per_label"].items():
                    wandb.run.summary[f"test_label_auprc/{lbl_name}"] = l_metrics["AUPRC"]
                    wandb.run.summary[f"test_label_auroc/{lbl_name}"] = l_metrics["AUROC"]
                    wandb.run.summary[f"test_label_f1/{lbl_name}"] = l_metrics["Calibrated_F1"]
                    wandb.run.summary[f"test_label_balanced_acc/{lbl_name}"] = l_metrics["Balanced_Accuracy"]

            wandb.finish()

        del model, optimizer, scheduler, criterion, train_loader, val_loader
        if test_loader is not None:
            del test_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Consolidated summary across all completed models
    print_final_summary(output_dir)


if __name__ == "__main__":
    main()
