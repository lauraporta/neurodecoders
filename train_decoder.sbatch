#!/bin/bash
#
#SBATCH -p gpu # partition (queue)
#SBATCH -J decoder_training
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 00-06:00 # time (D-HH:MM)
#SBATCH --array=0-2%3
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

export MLFLOW_TRACKING_URI=/ceph/margrie/laura/neurodecoders/mlruns
mkdir -p "$MLFLOW_TRACKING_URI"

export CUDA_VISIBLE_DEVICES=0

# ==========================================
# HYPERPARAMETER SWEEP CONFIGURATION:
# ==========================================
EPOCHS=100

# Define hyperparameter arrays
LEARNING_RATES=(1e-3)
BATCH_SIZES=(32)
MODEL_TYPES=("enhanced")
OPTIMIZERS=("adam")
LOSS_FUNCTIONS=("mse")
SCHEDULERS=("none")
DATASETS=("perlin_noise_patterns,70,70" "gabor,70,70" "periodic_patterns,70,70")

# Calculate indices from array task ID
# For this simple case, we're just sweeping over datasets
DATASET_IDX=$SLURM_ARRAY_TASK_ID

# Extract values
MODEL_TYPE=${MODEL_TYPES[0]}
LEARNING_RATE=${LEARNING_RATES[0]}
BATCH_SIZE=${BATCH_SIZES[0]}
OPTIMIZER=${OPTIMIZERS[0]}
LOSS_FUNCTION=${LOSS_FUNCTIONS[0]}
SCHEDULER=${SCHEDULERS[0]}
DATASET=${DATASETS[$DATASET_IDX]}

# Create run name with all hyperparameters
RUN_NAME="dataset=${DATASET}_model=${MODEL_TYPE}_lr=${LEARNING_RATE}_loss=${LOSS_FUNCTION}"
EXPERIMENT_NAME="decoder_dataset_sweep"

echo "Array Task $SLURM_ARRAY_TASK_ID: model=$MODEL_TYPE, lr=$LEARNING_RATE, batch_size=$BATCH_SIZE, optimizer=$OPTIMIZER, loss=$LOSS_FUNCTION, scheduler=$SCHEDULER, epochs=$EPOCHS, dataset=$DATASET"

# Pass individual parameters to the Python script
/nfs/nhome/live/lporta/.conda/envs/decoders/bin/python neurodecoders/decoder/mlflow_training.py \
    --model-type "$MODEL_TYPE" \
    --learning-rate "$LEARNING_RATE" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --optimizer "$OPTIMIZER" \
    --loss-function "$LOSS_FUNCTION" \
    --scheduler "$SCHEDULER" \
    --mixed-precision \
    --early-stopping \
    --checkpointing \
    --sta-type "$DATASET" \
    --n-neurons 100 \
    --n-images 8000 \
    --experiment-name "$EXPERIMENT_NAME" \
    --run-name "$RUN_NAME"

echo "Decoder training completed for task $SLURM_ARRAY_TASK_ID"
echo "MLflow experiments should be available in: ${MLFLOW_TRACKING_URI}"
echo "[INFO] End time: $(date)"
