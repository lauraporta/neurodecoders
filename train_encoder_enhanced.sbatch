#!/bin/bash
#
#SBATCH -p gpu_lowp,gpu # partition (queue)
#SBATCH -J encoder_enhanced_training
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 00-01:00 # time (D-HH:MM)
#SBATCH --array=0-1
#SBATCH -o /ceph/margrie/laura/neurodecoders/slurm/test_%j.out.txt
#SBATCH -e /ceph/margrie/laura/neurodecoders/slurm/test_%j.err.txt

echo "[INFO] Single encoder test run"
echo "[INFO] SLURM_JOB_ID=$SLURM_JOB_ID"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Start time: $(date)"

# Load modules and activate conda environment
source /etc/profile.d/modules.sh
module load miniconda
conda activate /nfs/nhome/live/lporta/.conda/envs/decoders
which python

# SLURM automatically sets CUDA_VISIBLE_DEVICES based on --gres allocation
# Do not manually override it to avoid conflicts


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
EPOCHS=100

# Define hyperparameter arrays
LEARNING_RATES=(1e-3)
BATCH_SIZES=(32)
MODEL_TYPES=("resnet_conv_only" "simple3layer")  # New spatial readout model
OPTIMIZERS=("adam")
SCHEDULERS=("plateau")  # Reduce LR on plateau for better convergence
DATASETS=("gabor,11,11")  # Small kernels for 32x32 images
DATASET_TYPES=("cifar10")  # Native 32x32 resolution

# ------------------------------------------
# Compute lengths and total combinations
# ------------------------------------------
NUM_LR=${#LEARNING_RATES[@]}
NUM_BS=${#BATCH_SIZES[@]}
NUM_MODEL_TYPES=${#MODEL_TYPES[@]}
NUM_OPTIMIZERS=${#OPTIMIZERS[@]}
NUM_SCHEDULERS=${#SCHEDULERS[@]}
NUM_DATASETS=${#DATASETS[@]}
NUM_DATASET_TYPES=${#DATASET_TYPES[@]}

# Define axis order, fastest varying first to preserve prior behavior
# Order: MODEL_TYPES -> DATASETS -> LEARNING_RATES -> BATCH_SIZES -> OPTIMIZERS -> SCHEDULERS
STRIDE_MODEL_TYPES=1
STRIDE_DATASETS=$((STRIDE_MODEL_TYPES * NUM_MODEL_TYPES))
STRIDE_LR=$((STRIDE_DATASETS * NUM_DATASETS))
STRIDE_BS=$((STRIDE_LR * NUM_LR))
STRIDE_OPTIMIZERS=$((STRIDE_BS * NUM_BS))
STRIDE_SCHEDULERS=$((STRIDE_OPTIMIZERS * NUM_OPTIMIZERS))
STRIDE_DATASET_TYPES=$((STRIDE_SCHEDULERS * NUM_SCHEDULERS))

TOTAL_COMBINATIONS=$((NUM_MODEL_TYPES * NUM_DATASETS * NUM_LR * NUM_BS * NUM_OPTIMIZERS * NUM_SCHEDULERS * NUM_DATASET_TYPES))

# Guard: if task id exceeds combinations, exit gracefully (supports oversize arrays)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID >= TOTAL_COMBINATIONS )); then
    echo "Task ID ${TASK_ID} exceeds total combinations ${TOTAL_COMBINATIONS}. Exiting gracefully."
    exit 0
fi

# Calculate indices from array task ID using mixed-radix decomposition
MODEL_TYPE_IDX=$(( (TASK_ID / STRIDE_MODEL_TYPES) % NUM_MODEL_TYPES ))
DATASET_IDX=$(( (TASK_ID / STRIDE_DATASETS) % NUM_DATASETS ))
LEARNING_RATE_IDX=$(( (TASK_ID / STRIDE_LR) % NUM_LR ))
BATCH_SIZE_IDX=$(( (TASK_ID / STRIDE_BS) % NUM_BS ))
OPTIMIZER_IDX=$(( (TASK_ID / STRIDE_OPTIMIZERS) % NUM_OPTIMIZERS ))
SCHEDULER_IDX=$(( (TASK_ID / STRIDE_SCHEDULERS) % NUM_SCHEDULERS ))
DATASET_TYPE_IDX=$(((TASK_ID / STRIDE_DATASET_TYPES) % NUM_DATASET_TYPES ))

# Extract values
MODEL_TYPE=${MODEL_TYPES[$MODEL_TYPE_IDX]}
LEARNING_RATE=${LEARNING_RATES[$LEARNING_RATE_IDX]}
BATCH_SIZE=${BATCH_SIZES[$BATCH_SIZE_IDX]}
OPTIMIZER=${OPTIMIZERS[$OPTIMIZER_IDX]}
SCHEDULER=${SCHEDULERS[$SCHEDULER_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}
DATASET_TYPE=${DATASET_TYPES[$DATASET_TYPE_IDX]}

# Create run name with all hyperparameters
RUN_NAME="model_${MODEL_TYPE}_dataset_${DATASET}_lr_${LEARNING_RATE}_sched_${SCHEDULER}_type_${DATASET_TYPE}_paper_approach"
EXPERIMENT_NAME="long_training_paper_approach"

echo "Array Task $TASK_ID of $TOTAL_COMBINATIONS: model=$MODEL_TYPE, lr=$LEARNING_RATE, batch_size=$BATCH_SIZE, optimizer=$OPTIMIZER, scheduler=$SCHEDULER, epochs=$EPOCHS, dataset=$DATASET, dataset_type=$DATASET_TYPE"

# Pass individual parameters to the Python script
/nfs/nhome/live/lporta/.conda/envs/decoders/bin/python neurodecoders/encoder/mlflow_training.py \
    --model-type "$MODEL_TYPE" \
    --learning-rate "$LEARNING_RATE" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --optimizer "$OPTIMIZER" \
    --loss-function mse \
    --scheduler "$SCHEDULER" \
    --mixed-precision \
    --no-early-stopping \
    --checkpointing \
    --sta-type "$DATASET" \
    --dataset-type "$DATASET_TYPE" \
    --n-neurons 5000 \
    --n-images 8000 \
    --cv-folds 1 \
    --experiment-name "$EXPERIMENT_NAME" \
    --run-name "$RUN_NAME" 


echo "Enhanced training completed for task $TASK_ID"
echo "MLflow experiments should be available in: ${MLFLOW_TRACKING_URI}"
