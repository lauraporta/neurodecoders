#!/bin/bash
#
#SBATCH -p gpu_lowp,gpu,a100 # partition (queue)
#SBATCH -J input_optim
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 00-01:00 # time (D-HH:MM)
#SBATCH -o /ceph/margrie/laura/neurodecoders/slurm/input_optim_%j.out.txt
#SBATCH -e /ceph/margrie/laura/neurodecoders/slurm/input_optim_%j.err.txt

echo "[INFO] Input optimization run"
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

# Optional overrides
MODEL_ID=${MODEL_ID:-m-55fdb61b173e4a6181fc6c92abebfab6}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-input_optimization}
STEPS=${STEPS:-3000}
LR=${LR:-0.001}
IMAGE_SIZE=${IMAGE_SIZE:-64}
CHANNELS=${CHANNELS:-1}
TV_WEIGHT=${TV_WEIGHT:-1e-4}
L2_WEIGHT=${L2_WEIGHT:-1e-6}
LOG_EVERY=${LOG_EVERY:-50}
SEED=${SEED:-42}
OUTPUT_DIR=${OUTPUT_DIR:-}
SAMPLE_INDEX=${SAMPLE_INDEX:-0}

# Image IDs for multiple reconstruction (set to empty for single image)
# Examples: IMAGE_IDS="0 1 2 10" or IMAGE_IDS="5 15 25" or IMAGE_IDS=""
IMAGE_IDS=${IMAGE_IDS:-"0 1 2 10"}

# Generate run name
if [[ -n "$IMAGE_IDS" ]]; then
  RUN_NAME=${RUN_NAME:-input_optim_${SLURM_JOB_ID}_imageIDs_${IMAGE_IDS// /_}_modelID_${MODEL_ID}}
else
  RUN_NAME=${RUN_NAME:-input_optim_${SLURM_JOB_ID}_sample_${SAMPLE_INDEX}_modelID_${MODEL_ID}}
fi

# Log configuration
echo "[INFO] MODEL_ID=$MODEL_ID"
echo "[INFO] EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "[INFO] RUN_NAME=$RUN_NAME"
if [[ -n "$IMAGE_IDS" ]]; then
  echo "[INFO] IMAGE_IDS=$IMAGE_IDS (multiple image reconstruction)"
else
  echo "[INFO] SAMPLE_INDEX=$SAMPLE_INDEX (single image reconstruction)"
fi

# Build command arguments
CMD_ARGS=(
  --model-id "$MODEL_ID"
  --tracking-uri "file:$MLFLOW_TRACKING_URI"
  --experiment-name "$EXPERIMENT_NAME"
  --run-name "$RUN_NAME"
  --steps "$STEPS"
  --lr "$LR"
  --image-size "$IMAGE_SIZE"
  --channels "$CHANNELS"
  --tv-weight "$TV_WEIGHT"
  --l2-weight "$L2_WEIGHT"
  --log-every "$LOG_EVERY"
  --seed "$SEED"
)

# Add image specification
if [[ -n "$IMAGE_IDS" ]]; then
  # Convert space-separated string to array and add to command
  IFS=' ' read -ra IMAGE_ARRAY <<< "$IMAGE_IDS"
  CMD_ARGS+=(--image-ids)
  CMD_ARGS+=("${IMAGE_ARRAY[@]}")
else
  CMD_ARGS+=(--sample-index "$SAMPLE_INDEX")
fi

# Add output directory if specified
if [[ -n "$OUTPUT_DIR" ]]; then
  CMD_ARGS+=(--output-dir "$OUTPUT_DIR")
fi

# Execute the command
/nfs/nhome/live/lporta/.conda/envs/decoders/bin/python \
  neurodecoders/input_optim/mlflow_run.py \
  "${CMD_ARGS[@]}"

echo "[INFO] Input optimization completed"
echo "[INFO] MLflow path: ${MLFLOW_TRACKING_URI}"
echo "[INFO] End time: $(date)"
