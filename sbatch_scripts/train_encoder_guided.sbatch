#!/bin/bash
#
# ============================================================
# ENCODER-GUIDED DECODER EXPERIMENT
# ============================================================
# Trains decoders using a frozen encoder in the loss computation.
# The loss is computed on firing rates, not pixels.
#
# Architecture:
#   firing_rates -> decoder (trainable) -> image -> encoder (frozen) -> predicted_rates
#
# USAGE:
#   Full sweep (8 tasks):   sbatch train_encoder_guided.sbatch
#   Single run only:        sbatch --array=0 train_encoder_guided.sbatch
#   Only simple decoders:   sbatch --array=0-3 train_encoder_guided.sbatch
#   Only transformer:       sbatch --array=4-7 train_encoder_guided.sbatch
#
# Total combinations: 2 decoder types × 4 loss types = 8 runs
# ============================================================
#
#SBATCH -p gpu_lowp,a100 
#SBATCH -J enc_guided
#SBATCH --array=0-7%4    # 8 tasks, max 4 concurrent
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 00-04:00      # 4 hours max
#SBATCH -o /ceph/margrie/laura/neurodecoders/slurm/enc_guided_%j.out.txt
#SBATCH -e /ceph/margrie/laura/neurodecoders/slurm/enc_guided_%j.err.txt
#SBATCH --exclude=gpu-sr670-21,gpu-sr670-22

echo "[INFO] Encoder-Guided Decoder Training"
echo "[INFO] SLURM_JOB_ID=$SLURM_JOB_ID"
echo "[INFO] SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Start time: $(date)"

# Load modules and activate conda environment
# Use conditional sourcing to handle different node configurations
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
module load miniconda 2>/dev/null || true

# Initialize conda properly
eval "$(conda shell.bash hook)"
conda activate /nfs/nhome/live/lporta/.conda/envs/decoders
which python

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Load environment variables from .env file
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "[INFO] Loaded environment variables from .env"
    echo "[INFO] MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"
else
    echo "[WARNING] .env file not found"
fi

# ==========================================
# FIXED PARAMETERS
# ==========================================
EPOCHS=150
BATCH_SIZE=32
LEARNING_RATE=1e-4
OPTIMIZER="adamw"
SCHEDULER="cosine"
WEIGHT_DECAY=1e-5

# Encoder to use (pre-trained, will be frozen)
ENCODER_RUN_ID="9c5b51243d6248bc8967f8c5d224572d"

# Dataset configuration (must match encoder training)
DATASET_TYPE="cifar10"
STA_TYPE="gabor,11,11"
N_NEURONS=5000
N_TRAIN_IMAGES=8000
N_TEST_IMAGES=2000

# ==========================================
# SWEEP CONFIGURATION
# ==========================================
# 2 decoder types × 4 loss types = 8 combinations
DECODER_TYPES=("simple" "transformer")
LOSS_TYPES=("mse" "poisson" "correlation" "combined")

# Calculate indices
task_id=${SLURM_ARRAY_TASK_ID:-0}
n_loss_types=${#LOSS_TYPES[@]}

decoder_idx=$((task_id / n_loss_types))
loss_idx=$((task_id % n_loss_types))

DECODER_TYPE=${DECODER_TYPES[$decoder_idx]}
LOSS_TYPE=${LOSS_TYPES[$loss_idx]}

# Generate run name
RUN_NAME="${DECODER_TYPE}_${LOSS_TYPE}"

echo "============================================"
echo "Configuration for task $task_id:"
echo "  Decoder type: $DECODER_TYPE"
echo "  Loss type: $LOSS_TYPE"
echo "  Encoder run ID: $ENCODER_RUN_ID"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Run name: $RUN_NAME"
echo "============================================"

# ==========================================
# RUN TRAINING
# ==========================================
cd /nfs/nhome/live/lporta/source/neurodecoders

python -m neurodecoders.decoder.encoder_guided_training \
    --encoder_run_id "$ENCODER_RUN_ID" \
    --base_decoder_type "$DECODER_TYPE" \
    --loss_type "$LOSS_TYPE" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --optimizer "$OPTIMIZER" \
    --scheduler "$SCHEDULER" \
    --weight_decay $WEIGHT_DECAY \
    --dataset_type "$DATASET_TYPE" \
    --sta_type "$STA_TYPE" \
    --n_neurons $N_NEURONS \
    --n_train_images $N_TRAIN_IMAGES \
    --n_test_images $N_TEST_IMAGES \
    --mlflow_experiment_name "encoder_guided_decoder" \
    --mlflow_run_name "$RUN_NAME" \
    --num_workers 4 \
    --enable_mixed_precision True \
    --enable_early_stopping True \
    --early_stopping_patience 30 \
    --log_images_every 10

exit_code=$?

echo "[INFO] End time: $(date)"
echo "[INFO] Exit code: $exit_code"

exit $exit_code
