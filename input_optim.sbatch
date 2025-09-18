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
CHANNELS=${CHANNELS:-1}
TV_WEIGHT=${TV_WEIGHT:-1e-4}
L2_WEIGHT=${L2_WEIGHT:-1e-6}
LOG_EVERY=${LOG_EVERY:-50}
SEED=${SEED:-42}
OUTPUT_DIR=${OUTPUT_DIR:-}
SAMPLE_INDEX=${SAMPLE_INDEX:-0}
IMAGE_IDS=${IMAGE_IDS:-}
# include the parameters used (lr, steps, tv_weight, l2_weight) in the run name
RUN_NAME=${RUN_NAME:-input_optim_${SLURM_JOB_ID}_imageIDs_${IMAGE_IDS:-sample_${SAMPLE_INDEX}}_modelID_${MODEL_ID}}
echo "[INFO] MODEL_ID=$MODEL_ID"
if [[ -n "$TARGET_RATES_NPY" ]]; then
  echo "[INFO] TARGET_RATES_NPY=$TARGET_RATES_NPY"
else
  echo "[INFO] TARGET_RATES_NPY not set; will infer from MLflow dataset"
  if [[ -n "$IMAGE_IDS" ]]; then
    echo "[INFO] IMAGE_IDS=$IMAGE_IDS (multiple image reconstruction)"
  else
    echo "[INFO] SAMPLE_INDEX=$SAMPLE_INDEX (single image reconstruction)"
  fi
fi
echo "[INFO] EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "[INFO] RUN_NAME=$RUN_NAME"

/nfs/nhome/live/lporta/.conda/envs/decoders/bin/python \
  neurodecoders/input_optim/mlflow_run.py \
  --model-id "$MODEL_ID" \
  --tracking-uri "file:$MLFLOW_TRACKING_URI" \
  --experiment-name "$EXPERIMENT_NAME" \
  --run-name "$RUN_NAME" \
  --steps "$STEPS" \
  --lr "$LR" \
  --image-size "$IMAGE_SIZE" \
  --channels "$CHANNELS" \
  --tv-weight "$TV_WEIGHT" \
  --l2-weight "$L2_WEIGHT" \
  --log-every "$LOG_EVERY" \
  --seed "$SEED" \
  ${IMAGE_IDS:+--image-ids $IMAGE_IDS} \
  ${IMAGE_IDS:---sample-index "$SAMPLE_INDEX"} \
  ${OUTPUT_DIR:+--output-dir "$OUTPUT_DIR"}

echo "[INFO] Input optimization completed"
echo "[INFO] MLflow path: ${MLFLOW_TRACKING_URI}"
echo "[INFO] End time: $(date)"
