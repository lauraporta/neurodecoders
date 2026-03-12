#!/bin/bash
#
#SBATCH -p gpu_lowp,gpu # partition (queue)
#SBATCH -J input_optim
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 00-00:20 # time (D-HH:MM)
#SBATCH --exclude=gpu-sr670-21  # Exclude nodes with CUDA driver issues
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
MODEL_IDS=("9c5b51243d6248bc8967f8c5d224572d") #"2f040db2f5bd42718b3fe123c554cd45" "afeba15f6eb04a8a9c754e9c913f1a5d") 
STEPS=1000  
IMAGE_SIZE=32  # Match encoder training size (CIFAR-10 resized to 32x32)

# Define hyperparameter arrays
LEARNING_RATES=(0.001) 
IMAGE_IDS=("250") # "0" "50" "100" "350" "400")  # Space-separated image IDs to reconstruct
SCHEDULERS=("none")  # Learning rate scheduler options
LOSS_FUNCTIONS=("poisson_mean")  # Loss function options
BLUR_SIGMAS=(0)  # Gaussian blur sigma values for gradient smoothing

# ------------------------------------------
# Compute lengths and total combinations
# ------------------------------------------
NUM_MODEL_IDS=${#MODEL_IDS[@]}
NUM_LR=${#LEARNING_RATES[@]}
NUM_IMG_IDS=${#IMAGE_IDS[@]}
NUM_SCHEDULERS=${#SCHEDULERS[@]}
NUM_LOSS_FUNCTIONS=${#LOSS_FUNCTIONS[@]}
NUM_BLUR_SIGMAS=${#BLUR_SIGMAS[@]}

# Define axis order, fastest varying first
# Order: LEARNING_RATES -> LOSS_FUNCTIONS -> BLUR_SIGMAS -> IMAGE_IDS -> SCHEDULERS -> MODEL_IDS
STRIDE_LR=1
STRIDE_LOSS_FUNCTIONS=$((STRIDE_LR * NUM_LR))
STRIDE_BLUR_SIGMAS=$((STRIDE_LOSS_FUNCTIONS * NUM_LOSS_FUNCTIONS))
STRIDE_IMG_IDS=$((STRIDE_BLUR_SIGMAS * NUM_BLUR_SIGMAS))
STRIDE_SCHEDULERS=$((STRIDE_IMG_IDS * NUM_IMG_IDS))
STRIDE_MODEL_IDS=$((STRIDE_SCHEDULERS * NUM_SCHEDULERS))

TOTAL_COMBINATIONS=$((NUM_LR * NUM_LOSS_FUNCTIONS * NUM_BLUR_SIGMAS * NUM_IMG_IDS * NUM_SCHEDULERS * NUM_MODEL_IDS))

# Guard: if task id exceeds combinations, exit gracefully (supports oversize arrays)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID >= TOTAL_COMBINATIONS )); then
    echo "Task ID ${TASK_ID} exceeds total combinations ${TOTAL_COMBINATIONS}. Exiting gracefully."
    exit 0
fi

# Calculate indices from array task ID using mixed-radix decomposition
LR_IDX=$(( (TASK_ID / STRIDE_LR) % NUM_LR ))
LOSS_FUNCTION_IDX=$(( (TASK_ID / STRIDE_LOSS_FUNCTIONS) % NUM_LOSS_FUNCTIONS ))
BLUR_SIGMA_IDX=$(( (TASK_ID / STRIDE_BLUR_SIGMAS) % NUM_BLUR_SIGMAS ))
IMG_IDS_IDX=$(( (TASK_ID / STRIDE_IMG_IDS) % NUM_IMG_IDS ))
SCHEDULER_IDX=$(( (TASK_ID / STRIDE_SCHEDULERS) % NUM_SCHEDULERS ))
MODEL_ID_IDX=$(( (TASK_ID / STRIDE_MODEL_IDS) % NUM_MODEL_IDS ))

# Extract values
LEARNING_RATE=${LEARNING_RATES[$LR_IDX]}
LOSS_FUNCTION=${LOSS_FUNCTIONS[$LOSS_FUNCTION_IDX]}
BLUR_SIGMA=${BLUR_SIGMAS[$BLUR_SIGMA_IDX]}
IMG_ID_SET=${IMAGE_IDS[$IMG_IDS_IDX]}
SCHEDULER=${SCHEDULERS[$SCHEDULER_IDX]}
MODEL_ID=${MODEL_IDS[$MODEL_ID_IDX]}

# Create run name with all hyperparameters
RUN_NAME="${MODEL_ID}_imgids_${IMG_ID_SET}_sched_${SCHEDULER}_lr_${LEARNING_RATE}_loss_${LOSS_FUNCTION}_blur_${BLUR_SIGMA}_images_sweep"
EXPERIMENT_NAME="input_optim_sweep_experiment"

echo "Array Task $TASK_ID of $TOTAL_COMBINATIONS: model_id=$MODEL_ID, lr=$LEARNING_RATE, loss=$LOSS_FUNCTION, blur_sigma=$BLUR_SIGMA, scheduler=$SCHEDULER, image_size=$IMAGE_SIZE, steps=$STEPS, image_ids=$IMG_ID_SET"

# Pass individual parameters to the Python script
/nfs/nhome/live/lporta/.conda/envs/decoders/bin/python neurodecoders/input_optim/mlflow_run.py \
    --model-id "$MODEL_ID" \
    --experiment-name "$EXPERIMENT_NAME" \
    --run-name "$RUN_NAME" \
    --steps "$STEPS" \
    --lr "$LEARNING_RATE" \
    --loss-function "$LOSS_FUNCTION" \
    --blur-sigma "$BLUR_SIGMA" \
    --scheduler "$SCHEDULER" \
    --image-size "$IMAGE_SIZE" \
    --channels 1 \
    --log-every 50 \
    --seed 42 \
    --image-ids $IMG_ID_SET

echo "Input optimization completed for task $TASK_ID"
echo "MLflow experiments should be available in: ${MLFLOW_TRACKING_URI}"
