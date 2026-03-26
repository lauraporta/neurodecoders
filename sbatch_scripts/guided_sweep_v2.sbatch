#!/bin/bash
#
# ============================================================
# GUIDED DECODER SWEEP V2 - Higher TV + Pixel Loss
# ============================================================
# Based on v1 findings:
#   - Simple decoder + correlation loss is the most stable config
#   - Higher TV weight improves pixel correlation (trend still rising at 0.015)
#   - Pixel loss adds direct image supervision to reduce patchiness
#
# GRID (9 jobs = 3 tv × 3 pixel_weight):
#   TV weights:
#     0 = 0.02
#     1 = 0.03
#     2 = 0.05
#   Pixel weights:
#     0 = 0.0   (neural-only baseline at higher TV)
#     1 = 0.1   (light pixel supervision)
#     2 = 0.5   (strong pixel supervision)
#
# All runs use: simple decoder, correlation loss
#
# USAGE:
#   sbatch --array=0-8 guided_sweep_v2.sbatch
#
# REQUIRES:
#   Trained encoder run_id saved at:
#   /ceph/margrie/laura/neurodecoders/workspace/encoder_run_id.txt
#
# ============================================================
#
#SBATCH -p gpu_lowp,a100
#SBATCH -J guided_v2
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 00-06:00
#SBATCH -o /ceph/margrie/laura/neurodecoders/slurm/guided_v2_%j_%a.out.txt
#SBATCH -e /ceph/margrie/laura/neurodecoders/slurm/guided_v2_%j_%a.err.txt
#SBATCH --exclude=gpu-sr670-21,gpu-sr670-22

# ============================================================
# ENVIRONMENT SETUP
# ============================================================

echo "[INFO] Guided Decoder Sweep V2"
echo "[INFO] SLURM_JOB_ID=$SLURM_JOB_ID"
echo "[INFO] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Start time: $(date)"

# Load modules and activate conda
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
module load miniconda 2>/dev/null || true

eval "$(conda shell.bash hook)"
conda activate /nfs/nhome/live/lporta/.conda/envs/decoders
which python

export CUDA_VISIBLE_DEVICES=0

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "[INFO] MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"
fi

# Create output directories
mkdir -p /ceph/margrie/laura/neurodecoders/workspace
mkdir -p /ceph/margrie/laura/neurodecoders/slurm

# ============================================================
# LOAD ENCODER RUN ID
# ============================================================
ENCODER_RUN_ID_FILE="/ceph/margrie/laura/neurodecoders/workspace/encoder_run_id.txt"

if [ ! -f "$ENCODER_RUN_ID_FILE" ]; then
    echo "[ERROR] Encoder run_id file not found: $ENCODER_RUN_ID_FILE"
    echo "[ERROR] Please run encoder training first:"
    echo "  sbatch --export=MODE=encoder architecture_comparison.sbatch"
    exit 1
fi
ENCODER_RUN_ID=$(cat "$ENCODER_RUN_ID_FILE")
echo "[INFO] Using encoder run_id: $ENCODER_RUN_ID"

# ============================================================
# GRID CONFIGURATION
# ============================================================

# TV weights (higher range, continuing from v1 trend)
TV_WEIGHTS=(0.02 0.03 0.05)

# Pixel loss weights (auxiliary image supervision)
PIXEL_WEIGHTS=(0.0 0.1 0.5)

# Grid indexing: task_id = tv_idx * 3 + pixel_idx
# Total: 3 tv × 3 pixel = 9 jobs (array 0-8)
TASK_ID=${SLURM_ARRAY_TASK_ID}

PIXEL_IDX=$((TASK_ID % 3))
TV_IDX=$((TASK_ID / 3))

# Extract parameters
TV_WEIGHT=${TV_WEIGHTS[$TV_IDX]}
PIXEL_WEIGHT=${PIXEL_WEIGHTS[$PIXEL_IDX]}

# Fixed: simple decoder + correlation loss (best from v1)
DECODER_TYPE="simple"
LOSS_TYPE="correlation"
LOSS_MSE=1.0
LOSS_CORR=0.1

# Common parameters
EPOCHS=200
LR=0.0001
BATCH_SIZE=32
OPTIMIZER="adam"
SCHEDULER="cosine"

# Dataset configuration
N_NEURONS=5000
N_TRAIN_IMAGES=8000
N_TEST_IMAGES=2000
DATASET_TYPE="cifar10"
STA_TYPE="gabor,11,11"
EXPERIMENT_NAME="architecture_comparison"

echo "[INFO] Grid position: tv_idx=$TV_IDX, pixel_idx=$PIXEL_IDX"
echo "[INFO] Configuration:"
echo "  Decoder type: $DECODER_TYPE"
echo "  Loss type: $LOSS_TYPE"
echo "  TV weight: $TV_WEIGHT"
echo "  Pixel weight: $PIXEL_WEIGHT"
echo "  Epochs: $EPOCHS"
echo "  LR: $LR"

# ============================================================
# RUN EXPERIMENT
# ============================================================

python -m neurodecoders.experiments.architecture_comparison \
    --mode guided \
    --encoder-run-id "$ENCODER_RUN_ID" \
    --n-neurons $N_NEURONS \
    --n-train-images $N_TRAIN_IMAGES \
    --n-test-images $N_TEST_IMAGES \
    --dataset-type $DATASET_TYPE \
    --sta-type $STA_TYPE \
    --experiment-name $EXPERIMENT_NAME \
    --guided-epochs $EPOCHS \
    --guided-lr $LR \
    --guided-batch-size $BATCH_SIZE \
    --guided-decoder-type $DECODER_TYPE \
    --guided-loss-type $LOSS_TYPE \
    --guided-loss-weights-mse $LOSS_MSE \
    --guided-loss-weights-corr $LOSS_CORR \
    --guided-tv-weight $TV_WEIGHT \
    --guided-pixel-weight $PIXEL_WEIGHT \
    --guided-optimizer $OPTIMIZER \
    --guided-scheduler $SCHEDULER

EXIT_CODE=$?

echo "[INFO] End time: $(date)"
echo "[INFO] Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] Task $TASK_ID completed: tv=$TV_WEIGHT / pixel=$PIXEL_WEIGHT"
else
    echo "[ERROR] Task $TASK_ID failed: tv=$TV_WEIGHT / pixel=$PIXEL_WEIGHT"
fi

exit $EXIT_CODE
