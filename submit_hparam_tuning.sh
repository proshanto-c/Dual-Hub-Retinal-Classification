#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8           # Request 8 CPU cores for dataloading
#SBATCH --mem=32G                   # Request 32GB total memory
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=47:59:00             # Max time limit
#SBATCH --job-name=hparam_dr
#SBATCH --array=1-8                 # Array tasks 0-8 run splits 0..8 (2 trials per worker = 18 total)
#SBATCH --output=logs/hparam_%A_%a.out
#SBATCH --error=logs/hparam_%A_%a.err

# 1. Activate Environment
source ~/miniconda/bin/activate brset-camlc

# 2. Ensure logs directory exists
mkdir -p logs

# 3. Setup Seed, Target, and Parameters
# Usage:
#   sbatch submit_hparam_tuning.sh [SEED] [TARGET]
# Examples:
#   sbatch submit_hparam_tuning.sh 42 dr            # Diabetic Retinopathy (middle prevalence, default)
#   sbatch submit_hparam_tuning.sh 42 mf            # Myopic Fundus (low prevalence)
#   sbatch submit_hparam_tuning.sh 42 multilabel    # Multi-label (all 7 conditions)
SEED=${1:-42}
TARGET=${2:-dr}                      # Options: 'dr' (default), 'mf', 'multilabel', or any disease name
SPLIT_TOTAL=9                        # 18 total trials / 9 splits = 2 trials per job

TARGET_LOWER=$(echo "$TARGET" | tr '[:upper:]' '[:lower:]')

case "$TARGET_LOWER" in
    dr|diabetic_retinopathy|singlelabel_dr)
        PROJECT="HParam-Tuning-Singlelabel-DR"
        OUTPUT_DIR="/users/sann7128/ca-mlc-refactor/data/BRSET/runs/hparam_dr_${SEED}"
        TARGET_FLAG="--target dr"
        ;;
    mf|myopic_fundus|singlelabel_mf|singlelabel_myopic_fundus|myopia)
        PROJECT="HParam-Tuning-Singlelabel-MF"
        OUTPUT_DIR="/users/sann7128/ca-mlc-refactor/data/BRSET/runs/hparam_mf_${SEED}"
        TARGET_FLAG="--target mf"
        ;;
    multi|multilabel)
        PROJECT="HParam-Tuning-Multilabel-ImgOnly"
        OUTPUT_DIR="/users/sann7128/ca-mlc-refactor/data/BRSET/runs/hparam_multi_${SEED}"
        TARGET_FLAG="--multilabel"
        ;;
    *)
        PROJECT="HParam-Tuning-Singlelabel-${TARGET}"
        OUTPUT_DIR="/users/sann7128/ca-mlc-refactor/data/BRSET/runs/hparam_${TARGET_LOWER}_${SEED}"
        TARGET_FLAG="--target ${TARGET_LOWER}"
        ;;
esac

# 4. Run Grid Search (Splits 0 to 8 across SLURM array tasks 0-8)
# If running split 0 locally, change array to 1-8
python tune_hyperparams.py \
    --root "/users/sann7128/ca-mlc-refactor/data/BRSET" \
    --output-dir "$OUTPUT_DIR" \
    --seed $SEED \
    --wandb-project "$PROJECT" \
    $TARGET_FLAG \
    --grid-search \
    --split-total $SPLIT_TOTAL \
    --split-index ${SLURM_ARRAY_TASK_ID:-0}
