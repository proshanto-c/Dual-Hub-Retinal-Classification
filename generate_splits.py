#!/usr/bin/env python3
import argparse
from pathlib import Path

# Import the modularized splitter
from data_functions.BRSETDataSplitter import BRSETDataSplitter

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate data splits for the BRSET dataset.")
    parser.add_argument(
        "--root", 
        type=Path, 
        required=True, 
        help="Path to the root directory containing 'labels_brset.csv' and 'fundus_photos/'"
    )
    parser.add_argument(
        "--split-basis", 
        type=str, 
        choices=["image", "patient"], 
        default="patient",
        help="Whether to calculate stratifications based on individual images or group by patient (default: patient)."
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="Proportion of data for training."
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="Proportion of data for validation."
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.15, help="Proportion of data for evaluation/testing."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--target-labels", nargs="*", default=[
            "increased_cup_disc", "drusens", "diabetic_retinopathy", 
            "macular_edema", "scar", "amd", "myopic_fundus"
        ], help="Target labels for patient-level multilabel stratification."
    )
    parser.add_argument(
        "--keep-inadequate", 
        action="store_true", 
        help="If flagged, includes images marked as inadequate quality."
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Ensure ratios sum to 1.0
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if round(total_ratio, 5) != 1.0:
        raise ValueError(f"Split ratios must sum to 1.0. Currently sum to {total_ratio}")

    print(f"Initializing BRSETDataSplitter...")
    print(f"  Root Directory: {args.root}")
    print(f"  Split Basis:    {args.split_basis.upper()}-level")
    print(f"  Ratios:         Train ({args.train_ratio}) | Val ({args.val_ratio}) | Test ({args.test_ratio})")
    print(f"  Seed:           {args.seed}")

    # 1. Instantiate the Splitter
    splitter = BRSETDataSplitter(
        root_dir=str(args.root),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        target_labels=args.target_labels,
        adequate_only=not args.keep_inadequate,
        split_basis=args.split_basis
    )

    # 2. Execute the splitting pipeline
    # (Assuming your class has a public wrapper method like split_data() 
    # that runs cohort filtering, splitting, and saving manifests)
    print("\nExecuting split pipeline...")
    splitter.split_data() 

    print("\n✅ Splitting complete!")
    print(f"Manifests and prevalence tables saved to: {splitter.output_directory}")

if __name__ == "__main__":
    main()