#!/bin/bash
#
#SBATCH -p gpu # partition (queue)
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
MODEL_TYPES=("enhanced" "mirror_simple")
OPTIMIZERS=("adam")
LOSS_FUNCTIONS=("mse" "l1")
SCHEDULERS=("none" "plateau")
DATASETS=("gabor,100,100" "periodic_patterns,100,100")

# Calculate indices from array task ID
# We have 2 model types × 2 loss × 2 sched × 2 datasets = 16 combos
# Update array range accordingly at top if needed (currently 0-7)
LOSS_IDX=$((SLURM_ARRAY_TASK_ID % 2))                 # 0 or 1
SCHED_IDX=$(((SLURM_ARRAY_TASK_ID / 2) % 2))          # 0 or 1
DATASET_IDX=$(((SLURM_ARRAY_TASK_ID / 4) % 2))        # 0 or 1
MODEL_IDX=$(((SLURM_ARRAY_TASK_ID / 8) % 2))          # 0 or 1

# Extract values
MODEL_TYPE=${MODEL_TYPES[$MODEL_IDX]}
LEARNING_RATE=${LEARNING_RATES[0]}
BATCH_SIZE=${BATCH_SIZES[0]}
OPTIMIZER=${OPTIMIZERS[0]}
LOSS_FUNCTION=${LOSS_FUNCTIONS[$LOSS_IDX]}
SCHEDULER=${SCHEDULERS[$SCHED_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}

# Create run name with all hyperparameters
# Extract dataset type from the dataset string (e.g., "gabor,100,100" -> "gabor")
DATASET_TYPE=$(echo "$DATASET" | cut -d',' -f1)
RUN_NAME="simplified_simulated_dataset_${DATASET_TYPE}_model=${MODEL_TYPE}_loss=${LOSS_FUNCTION}_sched=${SCHEDULER}"
EXPERIMENT_NAME="troubleshooting_overfitting2"

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
