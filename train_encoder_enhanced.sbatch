#!/bin/bash
#
#SBATCH -p gpu # partition (queue)
#SBATCH -J encoder_enhanced_training
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 00-06:00 # time (D-HH:MM)
#SBATCH --array=0-89%6 # 90 combinations (3 model types x 6 learning rates x 5 batch sizes), max 6 running simultaneously
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
EPOCHS=10000

# Define hyperparameter arrays
LEARNING_RATES=(1e-4 5e-4 1e-3 5e-3)
BATCH_SIZES=(8 16 32 64 128)
MODEL_TYPES=("resnet" "simple" "skip")

# Calculate indices from array task ID
TOTAL_LR=${#LEARNING_RATES[@]}
TOTAL_BS=${#BATCH_SIZES[@]}
TOTAL_MODELS=${#MODEL_TYPES[@]}

# Calculate which combination this array task should run
# Order: model_type, learning_rate, batch_size
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / (TOTAL_LR * TOTAL_BS)))
REMAINING=$((SLURM_ARRAY_TASK_ID % (TOTAL_LR * TOTAL_BS)))
LR_IDX=$((REMAINING / TOTAL_BS))
BS_IDX=$((REMAINING % TOTAL_BS))

# Extract values
MODEL_TYPE=${MODEL_TYPES[$MODEL_IDX]}
LEARNING_RATE=${LEARNING_RATES[$LR_IDX]}
BATCH_SIZE=${BATCH_SIZES[$BS_IDX]}

# Create run name
RUN_NAME="lr${LEARNING_RATE}_bs${BATCH_SIZE}_epochs${EPOCHS}_task${SLURM_ARRAY_TASK_ID}"
EXPERIMENT_NAME="second_run/${MODEL_TYPE}_encoder_comparison"

echo "Array Task $SLURM_ARRAY_TASK_ID: model=$MODEL_TYPE, lr=$LEARNING_RATE, batch_size=$BATCH_SIZE, epochs=$EPOCHS"

# Pass individual parameters to the Python script
/nfs/nhome/live/lporta/.conda/envs/decoders/bin/python neurodecoders/encoder/mlflow_training.py \
    --model-type "$MODEL_TYPE" \
    --learning-rate "$LEARNING_RATE" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --mixed-precision \
    --no-early-stopping \
    --checkpointing \

    --n-neurons 100 \
    --n-images 10000 \
    --cv-folds 1 \
    --experiment-name "$EXPERIMENT_NAME" \
    --run-name "$RUN_NAME"

echo "Enhanced training completed for task $SLURM_ARRAY_TASK_ID"
echo "MLflow experiments should be available in: ${MLFLOW_TRACKING_URI}"
