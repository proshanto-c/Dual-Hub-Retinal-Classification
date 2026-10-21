#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8           # Request 8 CPU cores for dataloading
#SBATCH --mem=32G                   # Request 32GB total memory
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=47:59:00             # Max time limit
#SBATCH --job-name=final_sweep
#SBATCH --array=0-4                 # Array tasks 0-4 run the 5 remaining labels in parallel
#SBATCH --output=logs/final_%A_%a.out
#SBATCH --error=logs/final_%A_%a.err

# 1. Activate Environment
source ~/miniconda/bin/activate brset-camlc

# 2. Ensure logs directory exists
mkdir -p logs

# 3. Setup Seed, Part, and Parameters
# Usage:
#   sbatch submit_final_sweep.sh [SEED] [PART]
# Examples:
#   sbatch submit_final_sweep.sh 42 part1       # Part 1: Candidate 1 (128h / 0.25d) across 5 labels (Array 0-4)
#   sbatch submit_final_sweep.sh 42 part2       # Part 2: Candidate 2 (512h / 0.15d) across 5 labels (Array 0-4)
#   sbatch --array=0-9 submit_final_sweep.sh 42 all  # Runs all 10 runs in parallel (Array 0-9)
SEED=${1:-42}
PART=${2:-part1}                     # Options: 'part1' (default), 'part2', or 'all'
PROJECT="Singlelabel-Image-Only-Final-Sweep"
OUTPUT_DIR="/users/sann7128/ca-mlc-refactor/data/BRSET/runs/final_sweep_${SEED}"

PART_LOWER=$(echo "$PART" | tr '[:upper:]' '[:lower:]')

case "$PART_LOWER" in
    1|part1)
        PART_ARG="--part 1"
        SPLIT_TOTAL=5                # 5 labels in Part 1 -> 1 label per job
        ;;
    2|part2)
        PART_ARG="--part 2"
        SPLIT_TOTAL=5                # 5 labels in Part 2 -> 1 label per job
        ;;
    all|both)
        PART_ARG="--part all"
        SPLIT_TOTAL=10               # 10 runs in total (5 labels x 2 candidates)
        ;;
    *)
        PART_ARG="--part $PART_LOWER"
        SPLIT_TOTAL=5
        ;;
esac

# 4. Run Sweep Runner
python sweep_remaining_singlelabels.py \
    --root "/users/sann7128/ca-mlc-refactor/data/BRSET" \
    --output-dir "$OUTPUT_DIR" \
    --seed $SEED \
    --wandb-project "$PROJECT" \
    $PART_ARG \
    --split-total $SPLIT_TOTAL \
    --split-index ${SLURM_ARRAY_TASK_ID:-0} \
    "${@:3}"
