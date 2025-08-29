#!/bin/bash
#
#SBATCH -p gpu # partition (queue)
#SBATCH -J encoder_enhanced_training
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 00-06:00 # time (D-HH:MM)
#SBATCH --array=0-3%3 
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
MODEL_TYPES=("resnet")
OPTIMIZERS=("adam")
SCHEDULERS=("none")
DATASETS=("perlin_noise_patterns,70,70", "gabor,70,70", "periodic_patterns,70,70")

# Calculate indices from array task ID
TOTAL_LR=${#LEARNING_RATES[@]}
TOTAL_BS=${#BATCH_SIZES[@]}
TOTAL_MODELS=${#MODEL_TYPES[@]}
TOTAL_OPT=${#OPTIMIZERS[@]}
TOTAL_SCH=${#SCHEDULERS[@]}
TOTAL_DATASETS=${#DATASETS[@]}

# Calculate which combination this array task should run
# Order: model_type, learning_rate, batch_size, optimizer, scheduler
TEMP1=$((SLURM_ARRAY_TASK_ID / (TOTAL_LR * TOTAL_BS * TOTAL_OPT * TOTAL_SCH)))
TEMP2=$((SLURM_ARRAY_TASK_ID % (TOTAL_LR * TOTAL_BS * TOTAL_OPT * TOTAL_SCH)))
TEMP3=$((TEMP2 / (TOTAL_BS * TOTAL_OPT * TOTAL_SCH)))
TEMP4=$((TEMP2 % (TOTAL_BS * TOTAL_OPT * TOTAL_SCH)))
TEMP5=$((TEMP4 / (TOTAL_OPT * TOTAL_SCH)))
TEMP6=$((TEMP4 % (TOTAL_OPT * TOTAL_SCH)))


MODEL_IDX=$TEMP1
LR_IDX=$TEMP3
BS_IDX=$((TEMP5))
OPT_IDX=$((TEMP6 / TOTAL_SCH))
SCH_IDX=$((TEMP6 % TOTAL_SCH))
DATASET_IDX=$((TEMP6 / TOTAL_DATASETS))

# Extract values
MODEL_TYPE=${MODEL_TYPES[$MODEL_IDX]}
LEARNING_RATE=${LEARNING_RATES[$LR_IDX]}
BATCH_SIZE=${BATCH_SIZES[$BS_IDX]}
OPTIMIZER=${OPTIMIZERS[$OPT_IDX]}
SCHEDULER=${SCHEDULERS[$SCH_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}

# Create run name with all hyperparameters
RUN_NAME="dataset=${DATASET}"
EXPERIMENT_NAME="dataset_sweep2"

echo "Array Task $SLURM_ARRAY_TASK_ID: model=$MODEL_TYPE, lr=$LEARNING_RATE, batch_size=$BATCH_SIZE, optimizer=$OPTIMIZER, scheduler=$SCHEDULER, epochs=$EPOCHS"

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
