#!/usr/bin/env bash

# Usage:
#   ./run_input_optim.sh --model-id m-123 --target-rates-npy /path/to/rates.npy \
#       [--tracking-uri file:/abs/path/mlruns] [--experiment-name input_optimization]
#       [--run-name my_run] [--steps 2000] [--lr 0.05]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"

MODEL_ID="m-553e4f38555b44a6a026362915f9431c"
TRACKING_URI=""
EXPERIMENT_NAME="input_optimization"
RUN_NAME=""
TARGET_RATES_NPY=""
STEPS=2000
LR=0.05
IMAGE_SIZE=64
CHANNELS=3
TV_WEIGHT=1e-4
L2_WEIGHT=1e-6
LOG_EVERY=50
SEED=42
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-id) MODEL_ID="$2"; shift 2 ;;
    --tracking-uri) TRACKING_URI="$2"; shift 2 ;;
    --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --target-rates-npy) TARGET_RATES_NPY="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --image-size) IMAGE_SIZE="$2"; shift 2 ;;
    --channels) CHANNELS="$2"; shift 2 ;;
    --tv-weight) TV_WEIGHT="$2"; shift 2 ;;
    --l2-weight) L2_WEIGHT="$2"; shift 2 ;;
    --log-every) LOG_EVERY="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$TARGET_RATES_NPY" ]]; then
  echo "--target-rates-npy is required"
  exit 1
fi

PYTHON=python3
ENTRY="neurodecoders/input_optim/mlflow_run.py"

CMD=("$PYTHON" "$ROOT_DIR/$ENTRY" \
  --model-id "$MODEL_ID" \
  --target-rates-npy "$TARGET_RATES_NPY" \
  --experiment-name "$EXPERIMENT_NAME" \
  --steps "$STEPS" \
  --lr "$LR" \
  --image-size "$IMAGE_SIZE" \
  --channels "$CHANNELS" \
  --tv-weight "$TV_WEIGHT" \
  --l2-weight "$L2_WEIGHT" \
  --log-every "$LOG_EVERY" \
  --seed "$SEED")

if [[ -n "$TRACKING_URI" ]]; then
  CMD+=(--tracking-uri "$TRACKING_URI")
fi
if [[ -n "$RUN_NAME" ]]; then
  CMD+=(--run-name "$RUN_NAME")
fi
if [[ -n "$OUTPUT_DIR" ]]; then
  CMD+=(--output-dir "$OUTPUT_DIR")
fi

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
