#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8           # Request 8 CPU cores for dataloading
#SBATCH --mem=32G                   # Request 32GB total memory
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=23:59:00             # Max time limit (fits standard 24h partitions)
#SBATCH --job-name=final_models
#SBATCH --array=0-20                # 21 cluster models (6 singlelabels + 1 multilabel) x 3 modalities
#SBATCH --output=logs/final_models_%A_%a.out
#SBATCH --error=logs/final_models_%A_%a.err

# ==============================================================================
# SLURM ARRAY SUBMISSION FOR FINAL CLUSTER MODELS (SINGLELABEL & MULTILABEL)
#
# Covers:
#   - drusens              (128h / 0.25d) (ctx_only, image_only, img_ctx)
#   - diabetic_retinopathy (512h / 0.15d) (ctx_only, image_only, img_ctx)
#   - increased_cup_disc   (512h / 0.15d) (ctx_only, image_only, img_ctx)
#   - macular_edema        (512h / 0.15d) (ctx_only, image_only, img_ctx)
#   - scar                 (512h / 0.15d) (ctx_only, image_only, img_ctx)
#   - amd                  (512h / 0.15d) (ctx_only, image_only, img_ctx)
#   - multilabel           (512h / 0.15d) (ctx_only, image_only, img_ctx)
#
# Usage:
#   sbatch submit_final_models.sh [SEED] [EXTRA_ARGS...]
#
# Examples:
#   # 1. Run all 21 cluster models for Seed 42 in parallel (Array 0-20):
#   sbatch submit_final_models.sh 42
#
#   # 2. Run across 3 different seeds (e.g. 42, 12, 3):
#   sbatch submit_final_models.sh 42
#   sbatch submit_final_models.sh 12
#   sbatch submit_final_models.sh 3
#
#   # 3. Run ONLY the 3 multilabel models (ctx_only, image_only, img_ctx):
#   sbatch --array=0-2 submit_final_models.sh 42 --labels multilabel
#
#   # 4. Run ONLY single-label cluster models (18 models across Array 0-17):
#   sbatch --array=0-17 submit_final_models.sh 42 --type singlelabel
#
#   # 5. Run ONLY drusen models (3 models across Array 0-2):
#   sbatch --array=0-2 submit_final_models.sh 42 --labels drusens
#
#   # 6. Run all cluster models for multimodal img_ctx only (7 models across Array 0-6):
#   sbatch --array=0-6 submit_final_models.sh 42 --modalities img_ctx
# ==============================================================================

# 1. Activate Environment
source ~/miniconda/bin/activate brset-camlc

# 2. Ensure logs directory exists
mkdir -p logs

# 3. Setup Seed and Paths
SEED=${1:-42}
PROJECT="Final-Fundus-Models"
OUTPUT_DIR="/users/sann7128/ca-mlc-refactor/data/BRSET/runs/final_models_${SEED}"

# If running within a SLURM job array, prioritize SLURM_ARRAY_TASK_COUNT
SPLIT_TOTAL=${SLURM_ARRAY_TASK_COUNT:-21}

# 4. Run Final Training Script for Cluster Tier (512h / 0.15d)
python train_final_models.py \
    --root "/users/sann7128/ca-mlc-refactor/data/BRSET" \
    --output-dir "$OUTPUT_DIR" \
    --seed $SEED \
    --wandb-project "$PROJECT" \
    --tier cluster \
    --split-total $SPLIT_TOTAL \
    --split-index ${SLURM_ARRAY_TASK_ID:-0} \
    "${@:2}"
