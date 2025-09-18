#!/bin/bash
#
#SBATCH -p gpu_lowp,gpu,a100 # partition (queue)
#SBATCH -J encoder_enhanced_training
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 00-01:00 # time (D-HH:MM)
#SBATCH --array=0-15%5
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
MODEL_TYPES=("simple" "skip" "resnet")
OPTIMIZERS=("adam")
SCHEDULERS=("none")
DATASETS=("gabor,70,70" "perlin_noise_patterns,100,100" "perlin_noise_patterns,70,70" "periodic_patterns,100,100" "periodic_patterns,70,70")

# Calculate indices from array task ID
# For this simple case, we're just sweeping over datasets
# DATASET_IDX=$SLURM_ARRAY_TASK_ID
MODEL_TYPE_IDX=$((SLURM_ARRAY_TASK_ID % 3))
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / 3))

# Extract values
MODEL_TYPE=${MODEL_TYPES[$MODEL_TYPE_IDX]}
LEARNING_RATE=${LEARNING_RATES[0]}
BATCH_SIZE=${BATCH_SIZES[0]}
OPTIMIZER=${OPTIMIZERS[0]}
SCHEDULER=${SCHEDULERS[0]}
DATASET=${DATASETS[$DATASET_IDX]}

# Create run name with all hyperparameters
RUN_NAME="model_${MODEL_TYPE}_Dataset_${DATASET}"
EXPERIMENT_NAME="new_datasets_sweep"

echo "Array Task $SLURM_ARRAY_TASK_ID: model=$MODEL_TYPE, lr=$LEARNING_RATE, batch_size=$BATCH_SIZE, optimizer=$OPTIMIZER, scheduler=$SCHEDULER, epochs=$EPOCHS, dataset=$DATASET"

# Pass individual parameters to the Python script
/nfs/nhome/live/lporta/.conda/envs/decoders/bin/python neurodecoders/encoder/mlflow_training.py \
    --model-type "$MODEL_TYPE" \
    --learning-rate "$LEARNING_RATE" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --optimizer "$OPTIMIZER" \
    --scheduler "$SCHEDULER" \
    --mixed-precision \
    --no-early-stopping \
    --checkpointing \
    --sta-type "$DATASET" \
    --n-neurons 100 \
    --n-images 8000 \
    --cv-folds 1 \
    --experiment-name "$EXPERIMENT_NAME" \
    --run-name "$RUN_NAME"

echo "Enhanced training completed for task $SLURM_ARRAY_TASK_ID"
echo "MLflow experiments should be available in: ${MLFLOW_TRACKING_URI}"
