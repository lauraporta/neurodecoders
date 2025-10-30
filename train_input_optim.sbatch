#!/bin/bash
#
#SBATCH -p gpu_lowp,gpu # partition (queue)
#SBATCH -J input_optim
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 00-01:00 # time (D-HH:MM)
#SBATCH --exclude=gpu-sr670-21  # Exclude nodes with CUDA driver issues
#SBATCH --array=0-3
#SBATCH -o /ceph/margrie/laura/neurodecoders/slurm/input_optim_%j.out.txt
#SBATCH -e /ceph/margrie/laura/neurodecoders/slurm/input_optim_%j.err.txt

echo "[INFO] Input Optimization (Image Reconstruction) run"
echo "[INFO] SLURM_JOB_ID=$SLURM_JOB_ID"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Start time: $(date)"

# Load modules and activate conda environment
source /etc/profile.d/modules.sh
module load miniconda
conda activate /nfs/nhome/live/lporta/.conda/envs/decoders
which python

export CUDA_VISIBLE_DEVICES=0

# Configure MLflow to use PostgreSQL database
export MLFLOW_TRACKING_URI="postgresql://laura:piscina@enc1-node9:5433/mlflow"

# ==========================================
# HYPERPARAMETER SWEEP CONFIGURATION:
# ==========================================

# Input optimization parameters
MODEL_ID="04d2e41a86d948acb30f0e8c3dad75cc"  # Run ID for CIFAR10 model
STEPS=100000  # Paper uses 1000 epochs

# Define hyperparameter arrays
# Paper settings: LR=1000 with gradient normalization + warmup, no explicit regularization
LEARNING_RATES=(0.001)  # Paper uses LR=1000 with gradient normalization
IMAGE_SIZES=(224)  # Match encoder training size (CIFAR-10 resized to 224x224)
IMAGE_IDS=("0")  # Space-separated image IDs to reconstruct

# ------------------------------------------
# Compute lengths and total combinations
# ------------------------------------------
NUM_LR=${#LEARNING_RATES[@]}
NUM_IMG_SIZE=${#IMAGE_SIZES[@]}
NUM_IMG_IDS=${#IMAGE_IDS[@]}

# Define axis order, fastest varying first
# Order: LEARNING_RATES -> IMAGE_SIZES -> IMAGE_IDS
STRIDE_LR=1
STRIDE_IMG_SIZE=$((STRIDE_LR * NUM_LR))
STRIDE_IMG_IDS=$((STRIDE_IMG_SIZE * NUM_IMG_SIZE))

TOTAL_COMBINATIONS=$((NUM_LR * NUM_IMG_SIZE * NUM_IMG_IDS))

# Guard: if task id exceeds combinations, exit gracefully (supports oversize arrays)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID >= TOTAL_COMBINATIONS )); then
    echo "Task ID ${TASK_ID} exceeds total combinations ${TOTAL_COMBINATIONS}. Exiting gracefully."
    exit 0
fi

# Calculate indices from array task ID using mixed-radix decomposition
LR_IDX=$(( (TASK_ID / STRIDE_LR) % NUM_LR ))
IMG_SIZE_IDX=$(( (TASK_ID / STRIDE_IMG_SIZE) % NUM_IMG_SIZE ))
IMG_IDS_IDX=$(( (TASK_ID / STRIDE_IMG_IDS) % NUM_IMG_IDS ))

# Extract values
LEARNING_RATE=${LEARNING_RATES[$LR_IDX]}
IMAGE_SIZE=${IMAGE_SIZES[$IMG_SIZE_IDX]}
IMG_ID_SET=${IMAGE_IDS[$IMG_IDS_IDX]}

# Create run name with all hyperparameters
RUN_NAME="input_optim_lr_${LEARNING_RATE}_size_${IMAGE_SIZE}"
EXPERIMENT_NAME="input_optim"

echo "Array Task $TASK_ID of $TOTAL_COMBINATIONS: lr=$LEARNING_RATE, image_size=$IMAGE_SIZE, steps=$STEPS, image_ids=$IMG_ID_SET"

# Pass individual parameters to the Python script
/nfs/nhome/live/lporta/.conda/envs/decoders/bin/python neurodecoders/input_optim/mlflow_run.py \
    --model-id "$MODEL_ID" \
    --experiment-name "$EXPERIMENT_NAME" \
    --run-name "$RUN_NAME" \
    --steps "$STEPS" \
    --lr "$LEARNING_RATE" \
    --image-size "$IMAGE_SIZE" \
    --channels 1 \
    --log-every 50 \
    --seed 42 \
    --image-ids $IMG_ID_SET

echo "Input optimization completed for task $TASK_ID"
echo "MLflow experiments should be available in: ${MLFLOW_TRACKING_URI}"
