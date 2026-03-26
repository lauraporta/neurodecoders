#!/bin/bash
#
# ============================================================
# ENCODER CONDITIONING EXPERIMENT FOR DIFFUSION DECODERS
# ============================================================
# USAGE:
#   sbatch encoder_conditioning_experiment.sbatch
#
# This script compares two conditioning approaches:
# 1. Baseline: Raw firing rates → simple MLP → conditioning
# 2. Encoder-Conditioned: Firing rates → learned mapper → encoder embedding → conditioning
# ============================================================
#
#SBATCH -p gpu_lowp,gpu # partition (queue)
#SBATCH -J enc_cond_exp
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 00-12:00 # time (D-HH:MM) - 12 hours for full training
#SBATCH --exclude=gpu-sr670-21,gpu-sr670-22  # Exclude nodes with CUDA driver issues
#SBATCH -o /ceph/margrie/laura/neurodecoders/slurm/enc_cond_%j.out.txt
#SBATCH -e /ceph/margrie/laura/neurodecoders/slurm/enc_cond_%j.err.txt

echo "[INFO] Encoder Conditioning Experiment"
echo "[INFO] SLURM_JOB_ID=$SLURM_JOB_ID"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Start time: $(date)"

# Load modules and activate conda environment
source /etc/profile.d/modules.sh
module load miniconda
conda activate /nfs/nhome/live/lporta/.conda/envs/decoders
which python

export CUDA_VISIBLE_DEVICES=0

# Load environment variables from .env file
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "[INFO] Loaded environment variables from .env"
    echo "[INFO] MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"
else
    echo "[WARNING] .env file not found, using default MLFLOW_TRACKING_URI"
fi

# ==========================================
# CONFIGURATION
# ==========================================

# Best encoder model from your training
ENCODER_MODEL_ID="9c5b51243d6248bc8967f8c5d224572d"

# Training parameters (matching best decoder config)
EPOCHS=100
MAPPER_EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=0.001
BASE_CHANNELS=64
TIMESTEPS=1000
INFERENCE_STEPS=50
N_TEST_SAMPLES=200

# MLflow settings
EXPERIMENT_NAME="encoder_conditioning_experiment"
RUN_NAME="enc_cond_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"

echo "============================================"
echo "ENCODER CONDITIONING EXPERIMENT CONFIGURATION"
echo "============================================"
echo "Encoder Model ID: $ENCODER_MODEL_ID"
echo "Epochs:           $EPOCHS"
echo "Mapper Epochs:    $MAPPER_EPOCHS"
echo "Batch Size:       $BATCH_SIZE"
echo "Learning Rate:    $LEARNING_RATE"
echo "Base Channels:    $BASE_CHANNELS"
echo "Timesteps:        $TIMESTEPS"
echo "Experiment:       $EXPERIMENT_NAME"
echo "Run Name:         $RUN_NAME"
echo "============================================"

# Run the experiment
/nfs/nhome/live/lporta/.conda/envs/decoders/bin/python \
    neurodecoders/decoder/encoder_conditioning_experiment.py \
    --encoder-model-id "$ENCODER_MODEL_ID" \
    --experiment-name "$EXPERIMENT_NAME" \
    --run-name "$RUN_NAME" \
    --epochs "$EPOCHS" \
    --mapper-epochs "$MAPPER_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --base-channels "$BASE_CHANNELS" \
    --timesteps "$TIMESTEPS" \
    --inference-steps "$INFERENCE_STEPS" \
    --n-test-samples "$N_TEST_SAMPLES" \
    --seed 42

EXIT_CODE=$?

echo "[INFO] Analysis completed with exit code: $EXIT_CODE"
echo "[INFO] End time: $(date)"
echo "[INFO] MLflow experiments available at: $MLFLOW_TRACKING_URI"
echo "[INFO] Check MLflow UI for plots and metrics under experiment: $EXPERIMENT_NAME"
