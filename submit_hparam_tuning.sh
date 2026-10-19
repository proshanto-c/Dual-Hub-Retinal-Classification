#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8           # Request 8 CPU cores for dataloading
#SBATCH --mem=32G                   # Request 32GB total memory
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=47:59:00             # Max time limit
#SBATCH --job-name=hparam_grid
#SBATCH --array=0-8                 # Array tasks 0-8 run splits 0..8 (2 trials per worker = 18 total)
#SBATCH --output=logs/hparam_%A_%a.out
#SBATCH --error=logs/hparam_%A_%a.err

# 1. Activate Environment
source ~/miniconda/bin/activate brset-camlc

# 2. Ensure logs directory exists
mkdir -p logs

# 3. Setup Seed and Parameters
SEED=${1:-42}
SPLIT_TOTAL=9                       # 18 total trials / 9 splits = 2 trials per job
PROJECT="HParam-Tuning-Multilabel-ImgOnly"
OUTPUT_DIR="/users/sann7128/ca-mlc-refactor/data/BRSET/runs/hparam_grid_${SEED}"

# 4. Run Grid Search (Splits 0 to 8 across SLURM array tasks 0-8)
# If running split 0 locally, change array to 1-8
python tune_hyperparams.py \
    --root "/users/sann7128/ca-mlc-refactor/data/BRSET" \
    --output-dir "$OUTPUT_DIR" \
    --seed $SEED \
    --wandb-project "$PROJECT" \
    --grid-search \
    --split-total $SPLIT_TOTAL \
    --split-index ${SLURM_ARRAY_TASK_ID:-0}
