#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8           # Request 8 CPU cores for dataloading
#SBATCH --mem=32G                   # Request 32GB total memory
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=23:59:00             # Max time limit (fits standard 24h partitions)
#SBATCH --job-name=resnet_fusion
#SBATCH --array=0-20                # 21 cluster models (6 singlelabels + 1 multilabel) x 3 modalities
#SBATCH --output=logs/resnet_fusion_%A_%a.out
#SBATCH --error=logs/resnet_fusion_%A_%a.err

# ==============================================================================
# SLURM ARRAY SUBMISSION FOR RESNET-50 + CLINICAL MLP BASELINE
#
# Trains and evaluates standard multimodal late fusion across identical splits,
# loss, inputs, and evaluation protocols to compare against the proposed GNN.
#
# Usage:
#   sbatch submit_resnet_fusion.sh [SEED] [EXTRA_ARGS...]
#
# Examples:
#   # 1. Run all 21 models for Seed 42 in parallel (Array 0-20):
#   sbatch submit_resnet_fusion.sh 42
#
#   # 2. Run across multiple seeds:
#   sbatch submit_resnet_fusion.sh 42
#   sbatch submit_resnet_fusion.sh 12
#   sbatch submit_resnet_fusion.sh 3
#   sbatch submit_resnet_fusion.sh 4
#   sbatch submit_resnet_fusion.sh 5
#
#   # 3. Run ONLY Multilabel models (ctx_only, image_only, img_ctx):
#   sbatch --array=0-2 submit_resnet_fusion.sh 42 --labels multilabel
#
#   # 4. Run ONLY multimodal img_ctx (7 models across Array 0-6):
#   sbatch --array=0-6 submit_resnet_fusion.sh 42 --modalities img_ctx
# ==============================================================================

# 1. Activate Environment
source ~/miniconda/bin/activate brset-camlc

# 2. Ensure logs directory exists
mkdir -p logs

# 3. Setup Seed and Paths
SEED=${1:-42}
PROJECT="ResNet-Fusion-Baseline"
OUTPUT_DIR="/users/sann7128/ca-mlc-refactor/data/BRSET/runs/resnet_fusion_${SEED}"

# If running within a SLURM job array, prioritize SLURM_ARRAY_TASK_COUNT
SPLIT_TOTAL=${SLURM_ARRAY_TASK_COUNT:-21}

# 4. Run Baseline Training Script
python train_resnet_fusion.py \
    --root "/users/sann7128/ca-mlc-refactor/data/BRSET" \
    --output-dir "$OUTPUT_DIR" \
    --seed $SEED \
    --wandb-project "$PROJECT" \
    --tier cluster \
    --split-total $SPLIT_TOTAL \
    --split-index ${SLURM_ARRAY_TASK_ID:-0} \
    "${@:2}"
