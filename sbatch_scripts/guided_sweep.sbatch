#!/bin/bash
#
# ============================================================
# GUIDED DECODER SWEEP - Grid Search for Improvements
# ============================================================
# Sweeps over loss type, TV weight, and decoder architecture
# to find the best guided decoder configuration.
#
# GRID (30 jobs total = 3 loss × 5 tv × 2 decoder):
#   Loss types:
#     0 = correlation
#     1 = combined (mse=0.5, corr=1.0)
#     2 = combined (mse=1.0, corr=1.0)
#   TV weights:
#     0 = 0.005
#     1 = 0.008
#     2 = 0.01
#     3 = 0.012
#     4 = 0.015
#   Decoder types:
#     0 = simple (CNN)
#     1 = transformer
#
# USAGE:
#   # Run full grid (30 jobs)
#   sbatch --array=0-29 guided_sweep.sbatch
#
#   # Run a subset (e.g., first 10)
#   sbatch --array=0-9 guided_sweep.sbatch
#
#   # Single job for testing
#   sbatch --array=0 guided_sweep.sbatch
#
# REQUIRES:
#   Trained encoder run_id saved at:
#   /ceph/margrie/laura/neurodecoders/workspace/encoder_run_id.txt
#
# ============================================================
#
#SBATCH -p gpu_lowp,a100
#SBATCH -J guided_sweep
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 00-06:00
#SBATCH -o /ceph/margrie/laura/neurodecoders/slurm/guided_sweep_%j_%a.out.txt
#SBATCH -e /ceph/margrie/laura/neurodecoders/slurm/guided_sweep_%j_%a.err.txt
#SBATCH --exclude=gpu-sr670-21,gpu-sr670-22

# ============================================================
# ENVIRONMENT SETUP
# ============================================================

echo "[INFO] Guided Decoder Sweep"
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

# Loss configurations: (loss_type, loss_weights_mse, loss_weights_corr)
LOSS_TYPES=("correlation" "combined" "combined")
LOSS_MSE_WEIGHTS=(1.0 0.5 1.0)
LOSS_CORR_WEIGHTS=(0.1 1.0 1.0)

# TV weights
TV_WEIGHTS=(0.005 0.008 0.01 0.012 0.015)

# Decoder types
DECODER_TYPES=("simple" "transformer")

# Grid indexing: task_id = loss_idx * 10 + tv_idx * 2 + decoder_idx
# Total: 3 loss × 5 tv × 2 decoder = 30 jobs (array 0-29)
TASK_ID=${SLURM_ARRAY_TASK_ID}

DECODER_IDX=$((TASK_ID % 2))
TV_IDX=$(( (TASK_ID / 2) % 5 ))
LOSS_IDX=$(( TASK_ID / 10 ))

# Extract parameters
LOSS_TYPE=${LOSS_TYPES[$LOSS_IDX]}
LOSS_MSE=${LOSS_MSE_WEIGHTS[$LOSS_IDX]}
LOSS_CORR=${LOSS_CORR_WEIGHTS[$LOSS_IDX]}
TV_WEIGHT=${TV_WEIGHTS[$TV_IDX]}
DECODER_TYPE=${DECODER_TYPES[$DECODER_IDX]}

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

echo "[INFO] Grid position: loss_idx=$LOSS_IDX, tv_idx=$TV_IDX, decoder_idx=$DECODER_IDX"
echo "[INFO] Configuration:"
echo "  Loss type: $LOSS_TYPE"
echo "  Loss weights: mse=$LOSS_MSE, corr=$LOSS_CORR"
echo "  TV weight: $TV_WEIGHT"
echo "  Decoder type: $DECODER_TYPE"
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
    --guided-optimizer $OPTIMIZER \
    --guided-scheduler $SCHEDULER

EXIT_CODE=$?

echo "[INFO] End time: $(date)"
echo "[INFO] Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] Task $TASK_ID completed: $DECODER_TYPE / $LOSS_TYPE / tv=$TV_WEIGHT"
else
    echo "[ERROR] Task $TASK_ID failed: $DECODER_TYPE / $LOSS_TYPE / tv=$TV_WEIGHT"
fi

exit $EXIT_CODE
