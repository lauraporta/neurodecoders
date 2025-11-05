#!/bin/bash
#
#SBATCH -p gpu_lowp,gpu # partition (queue)
#SBATCH -J input_optim
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 00-01:00 # time (D-HH:MM)
#SBATCH --exclude=gpu-sr670-21  # Exclude nodes with CUDA driver issues
#SBATCH --array=0-26
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

# Load environment variables from .env file
if [ -f .env ]; then
    set -a  # automatically export all variables
    source .env
    set +a
    echo "[INFO] Loaded environment variables from .env"
    echo "[INFO] MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"
else
    echo "[WARNING] .env file not found, using default MLFLOW_TRACKING_URI"
fi

# ==========================================
# HYPERPARAMETER SWEEP CONFIGURATION:
# ==========================================

# Input optimization parameters
MODEL_IDS=("72df8c56175c4d7a9381b0e3a5478e65" "d38f93de27854e21a61ae52c9ee25bbe" "8bfd15aefa4944de81561203cb25bf01") 
STEPS=2000  
IMAGE_SIZE=32  # Match encoder training size (CIFAR-10 resized to 32x32)

# Define hyperparameter arrays
LEARNING_RATES=(0.001)  # Paper uses LR=1000 with gradient normalization
IMAGE_IDS=("0" "50" "100" "150" "250" "350" "400")  # Space-separated image IDs to reconstruct

# ------------------------------------------
# Compute lengths and total combinations
# ------------------------------------------
NUM_MODEL_IDS=${#MODEL_IDS[@]}
NUM_LR=${#LEARNING_RATES[@]}
NUM_IMG_IDS=${#IMAGE_IDS[@]}

# Define axis order, fastest varying first
# Order: LEARNING_RATES -> IMAGE_IDS -> MODEL_IDS
STRIDE_LR=1
STRIDE_IMG_IDS=$((STRIDE_LR * NUM_LR))
STRIDE_MODEL_IDS=$((STRIDE_IMG_IDS * NUM_IMG_IDS))

TOTAL_COMBINATIONS=$((NUM_LR * NUM_IMG_IDS * NUM_MODEL_IDS))

# Guard: if task id exceeds combinations, exit gracefully (supports oversize arrays)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID >= TOTAL_COMBINATIONS )); then
    echo "Task ID ${TASK_ID} exceeds total combinations ${TOTAL_COMBINATIONS}. Exiting gracefully."
    exit 0
fi

# Calculate indices from array task ID using mixed-radix decomposition
LR_IDX=$(( (TASK_ID / STRIDE_LR) % NUM_LR ))
IMG_IDS_IDX=$(( (TASK_ID / STRIDE_IMG_IDS) % NUM_IMG_IDS ))
MODEL_ID_IDX=$(( (TASK_ID / STRIDE_MODEL_IDS) % NUM_MODEL_IDS ))

# Extract values
LEARNING_RATE=${LEARNING_RATES[$LR_IDX]}
IMG_ID_SET=${IMAGE_IDS[$IMG_IDS_IDX]}
MODEL_ID=${MODEL_IDS[$MODEL_ID_IDX]}

# Create run name with all hyperparameters
RUN_NAME="${MODEL_ID}_imgids_${IMG_ID_SET}"
EXPERIMENT_NAME="input_optim"

echo "Array Task $TASK_ID of $TOTAL_COMBINATIONS: model_id=$MODEL_ID, lr=$LEARNING_RATE, image_size=$IMAGE_SIZE, steps=$STEPS, image_ids=$IMG_ID_SET"

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
