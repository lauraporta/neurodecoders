#!/bin/bash
#
#SBATCH -p gpu_lowp,gpu,a100 # partition (queue)
#SBATCH -J decoder_training
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 00-01:00 # time (D-HH:MM)
#SBATCH --array=0-7
#SBATCH -o /ceph/margrie/laura/neurodecoders/slurm/decoder_%j.out.txt
#SBATCH -e /ceph/margrie/laura/neurodecoders/slurm/decoder_%j.err.txt

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

# Define hyperparameter arrays (aligned with paper)
LEARNING_RATES=(1e-3)
BATCH_SIZES=(32)
MODEL_TYPES=("simple" "mirror_simple")
OPTIMIZERS=("adam")
LOSS_FUNCTIONS=("mse" "l1")
SCHEDULERS=("none" "plateau")
DATASETS=("gabor,11,11")  # Match encoder: 11x11 Gabor kernels
DATASET_TYPES=("cifar10") # Match encoder: CIFAR-10 (32x32 native resolution)
N_NEURONS=5000            # Match encoder
N_TRAIN_IMAGES=8000       # Match encoder
N_TEST_IMAGES=2000        # Match encoder

# Calculate indices from array task ID
# We have 2 model types × 2 loss × 2 sched = 8 combos
LOSS_IDX=$((SLURM_ARRAY_TASK_ID % 2))                 # 0 or 1
SCHED_IDX=$(((SLURM_ARRAY_TASK_ID / 2) % 2))          # 0 or 1
MODEL_IDX=$(((SLURM_ARRAY_TASK_ID / 4) % 2))          # 0 or 1

# Extract values
MODEL_TYPE=${MODEL_TYPES[$MODEL_IDX]}
LEARNING_RATE=${LEARNING_RATES[0]}
BATCH_SIZE=${BATCH_SIZES[0]}
OPTIMIZER=${OPTIMIZERS[0]}
LOSS_FUNCTION=${LOSS_FUNCTIONS[$LOSS_IDX]}
SCHEDULER=${SCHEDULERS[$SCHED_IDX]}
DATASET=${DATASETS[0]}
DATASET_TYPE=${DATASET_TYPES[0]}

# Create run name with all hyperparameters
STA_TYPE=$(echo "$DATASET" | cut -d',' -f1)
RUN_NAME="model=${MODEL_TYPE}_loss=${LOSS_FUNCTION}_sched=${SCHEDULER}_sta=${STA_TYPE}_data=${DATASET_TYPE}"
EXPERIMENT_NAME="cifar10_decoding"

echo "Array Task $SLURM_ARRAY_TASK_ID: model=$MODEL_TYPE, lr=$LEARNING_RATE, batch_size=$BATCH_SIZE, optimizer=$OPTIMIZER, loss=$LOSS_FUNCTION, scheduler=$SCHEDULER, epochs=$EPOCHS, sta=$DATASET, dataset=$DATASET_TYPE, neurons=$N_NEURONS, train_imgs=$N_TRAIN_IMAGES, test_imgs=$N_TEST_IMAGES"

# Pass individual parameters to the Python script
/nfs/nhome/live/lporta/.conda/envs/decoders/bin/python neurodecoders/decoder/mlflow_training.py \
    --model-type "$MODEL_TYPE" \
    --learning-rate "$LEARNING_RATE" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --optimizer "$OPTIMIZER" \
    --loss-function "$LOSS_FUNCTION" \
    --scheduler "$SCHEDULER" \
    --image-size 32 \
    --mixed-precision \
    --early-stopping \
    --checkpointing \
    --sta-type "$DATASET" \
    --dataset-type "$DATASET_TYPE" \
    --n-neurons "$N_NEURONS" \
    --n-train-images "$N_TRAIN_IMAGES" \
    --n-test-images "$N_TEST_IMAGES" \
    --experiment-name "$EXPERIMENT_NAME" \
    --run-name "$RUN_NAME"

echo "Decoder training completed for task $SLURM_ARRAY_TASK_ID"
echo "MLflow experiments should be available in: ${MLFLOW_TRACKING_URI}"
echo "[INFO] End time: $(date)"
