#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8           # Request 8 CPU cores for dataloading
#SBATCH --mem=32G                   # Request 32GB total memory
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=47:59:00             # Max time limit
#SBATCH --job-name=final_models
#SBATCH --array=0-17                # 18 cluster models (5 singlelabels + 1 multilabel) x 3 modalities
#SBATCH --output=logs/final_models_%A_%a.out
#SBATCH --error=logs/final_models_%A_%a.err

# ==============================================================================
# SLURM ARRAY SUBMISSION FOR FINAL 512H CLUSTER MODELS (SINGLELABEL & MULTILABEL)
#
# Covers:
#   - diabetic_retinopathy (ctx_only, image_only, img_ctx)
#   - increased_cup_disc   (ctx_only, image_only, img_ctx)
#   - macular_edema        (ctx_only, image_only, img_ctx)
#   - scar                 (ctx_only, image_only, img_ctx)
#   - amd                  (ctx_only, image_only, img_ctx)
#   - multilabel           (ctx_only, image_only, img_ctx)
#
# Usage:
#   sbatch submit_final_models.sh [SEED] [EXTRA_ARGS...]
#
# Examples:
#   # 1. Run all 18 cluster models in parallel (Array 0-17):
#   sbatch submit_final_models.sh 42
#
#   # 2. Run ONLY the 3 multilabel models (ctx_only, image_only, img_ctx):
#   sbatch --array=0-2 submit_final_models.sh 42 --labels multilabel
#
#   # 3. Run ONLY single-label cluster models (15 models across Array 0-14):
#   sbatch --array=0-14 submit_final_models.sh 42 --type singlelabel
#
#   # 4. Run all cluster models for multimodal img_ctx only (6 models):
#   sbatch --array=0-5 submit_final_models.sh 42 --modalities img_ctx
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
SPLIT_TOTAL=${SLURM_ARRAY_TASK_COUNT:-18}

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
