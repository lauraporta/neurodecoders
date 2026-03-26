#!/bin/bash
#
# ============================================================
# DECODER HYPERPARAMETER SWEEP
# ============================================================
# USAGE:
#   Full sweep (48 tasks):   sbatch train_decoder.sbatch
#   Single run only:         sbatch --array=0 train_decoder.sbatch
#   Subset (simple models):  sbatch --array=0,3,6,9,12,15 train_decoder.sbatch
#
# Total combinations: 3 models × 2 LRs × 2 losses × 2 opts × 2 scheds = 48 runs
# Max concurrent jobs: 8 (set via %8 throttle)
# Estimated time per run: ~30-60 min (simple), varies for transformer/diffusion
# ============================================================
#
#SBATCH -p gpu_lowp,a100 # partition (queue)
#SBATCH -J decoder_sweep
#SBATCH --array=0-47%8   # 48 tasks, max 8 concurrent
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 00-02:00 # time (D-HH:MM) - increased for transformer/diffusion
#SBATCH -o /ceph/margrie/laura/neurodecoders/slurm/decoder_%j.out.txt
#SBATCH -e /ceph/margrie/laura/neurodecoders/slurm/decoder_%j.err.txt
#SBATCH --exclude=gpu-sr670-22

echo "[INFO] Decoder training run"
echo "[INFO] SLURM_JOB_ID=$SLURM_JOB_ID"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Start time: $(date)"

# Load modules and activate conda environment
source /etc/profile.d/modules.sh
module load miniconda
conda activate /nfs/nhome/live/lporta/.conda/envs/decoders
which python

# Load MLFLOW_TRACKING_URI from .env file (same as encoder)
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi
mkdir -p "$MLFLOW_TRACKING_URI"

# SLURM automatically sets CUDA_VISIBLE_DEVICES based on --gres allocation
# Do not manually override it to avoid conflicts

# ==========================================
# HYPERPARAMETER SWEEP CONFIGURATION:
# ==========================================
EPOCHS=100

# Define hyperparameter arrays for comprehensive sweep
# Total: 3 models × 2 LRs × 2 losses × 2 opts × 2 scheds = 48 combinations
LEARNING_RATES=(1e-3 1e-4)
BATCH_SIZES=(32)                  # Fixed - decoder not very sensitive to batch size
MODEL_TYPES=("simple" "transformer" "diffusion")
OPTIMIZERS=("adam" "adamw")
LOSS_FUNCTIONS=("mse" "smooth_l1")
SCHEDULERS=("none" "cosine")
DATASETS=("gabor,11,11")  # Match encoder: 11x11 Gabor kernels
DATASET_TYPES=("cifar10") # Match encoder: CIFAR-10 (32x32 native resolution)
N_NEURONS=5000            # Match encoder
N_TRAIN_IMAGES=8000       # Match encoder
N_TEST_IMAGES=2000        # Match encoder

# Calculate indices from array task ID
# Sweep order: model -> lr -> loss -> optimizer -> scheduler
# Total: 3 × 2 × 2 × 2 × 2 = 48 combinations
N_MODELS=${#MODEL_TYPES[@]}
N_LRS=${#LEARNING_RATES[@]}
N_LOSSES=${#LOSS_FUNCTIONS[@]}
N_OPTS=${#OPTIMIZERS[@]}
N_SCHEDS=${#SCHEDULERS[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

MODEL_IDX=$((TASK_ID % N_MODELS))
LR_IDX=$(((TASK_ID / N_MODELS) % N_LRS))
LOSS_IDX=$(((TASK_ID / (N_MODELS * N_LRS)) % N_LOSSES))
OPT_IDX=$(((TASK_ID / (N_MODELS * N_LRS * N_LOSSES)) % N_OPTS))
SCHED_IDX=$(((TASK_ID / (N_MODELS * N_LRS * N_LOSSES * N_OPTS)) % N_SCHEDS))

# Extract values
MODEL_TYPE=${MODEL_TYPES[$MODEL_IDX]}
LEARNING_RATE=${LEARNING_RATES[$LR_IDX]}
BATCH_SIZE=${BATCH_SIZES[0]}  # Fixed batch size
OPTIMIZER=${OPTIMIZERS[$OPT_IDX]}
LOSS_FUNCTION=${LOSS_FUNCTIONS[$LOSS_IDX]}
SCHEDULER=${SCHEDULERS[$SCHED_IDX]}
DATASET=${DATASETS[0]}
DATASET_TYPE=${DATASET_TYPES[0]}

# Create run name with all hyperparameters
STA_TYPE=$(echo "$DATASET" | cut -d',' -f1)
RUN_NAME="m=${MODEL_TYPE}_lr=${LEARNING_RATE}_bs=${BATCH_SIZE}_opt=${OPTIMIZER}_loss=${LOSS_FUNCTION}_sched=${SCHEDULER}"
EXPERIMENT_NAME="cifar10_decoding_sweep_normalized"

echo "=============================================="
echo "Array Task $TASK_ID / 47 (0-indexed)"
echo "  model=$MODEL_TYPE"
echo "  lr=$LEARNING_RATE"
echo "  batch_size=$BATCH_SIZE"
echo "  optimizer=$OPTIMIZER"
echo "  loss=$LOSS_FUNCTION"
echo "  scheduler=$SCHEDULER"
echo "  epochs=$EPOCHS"
echo "  dataset=$DATASET_TYPE, sta=$DATASET"
echo "  neurons=$N_NEURONS, train=$N_TRAIN_IMAGES, test=$N_TEST_IMAGES"
echo "=============================================="

# Build model-specific arguments
MODEL_ARGS=""
if [ "$MODEL_TYPE" == "transformer" ]; then
    MODEL_ARGS="--patch-size 4 --embed-dim 256 --num-heads 8 --num-layers 6"
elif [ "$MODEL_TYPE" == "diffusion" ]; then
    MODEL_ARGS="--base-channels 64 --channel-mults 1,2,4 --timesteps 1000 --inference-steps 50"
fi

# Pass individual parameters to the Python script
/nfs/nhome/live/lporta/.conda/envs/decoders/bin/python neurodecoders/decoder/mlflow_training.py \
    --model-type "$MODEL_TYPE" \
    --learning-rate "$LEARNING_RATE" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --optimizer "$OPTIMIZER" \
    --loss-function "$LOSS_FUNCTION" \
    --scheduler "$SCHEDULER" \
    --early-stopping \
    --checkpointing \
    --sta-type "$DATASET" \
    --dataset-type cifar10 \
    --n-neurons "$N_NEURONS" \
    --n-train-images "$N_TRAIN_IMAGES" \
    --n-test-images "$N_TEST_IMAGES" \
    --experiment-name "$EXPERIMENT_NAME" \
    --run-name "$RUN_NAME" \
    $MODEL_ARGS

echo "Decoder training completed for task $SLURM_ARRAY_TASK_ID"
echo "MLflow experiments should be available in: ${MLFLOW_TRACKING_URI}"
echo "[INFO] End time: $(date)"
