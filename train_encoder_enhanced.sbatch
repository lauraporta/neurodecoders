#!/bin/bash
#
# Enhanced Encoder Training - Architecture Comparison + Hyperparameter Sweep
#
# This script tests 3 different encoder architectures:
# - ResNet encoder (with frozen backbone)
# - Simple encoder (basic CNN)
# - Skip connection encoder (with residual connections)
#
# For each architecture, it tests:
# - 6 learning rates: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
# - 5 batch sizes: [8, 16, 32, 64, 128]
#
# Total: 3 × 6 × 5 = 90 combinations
#
# MLflow Organization:
# - Experiments: 3 separate experiments (one per architecture)
#   * resnet_encoder_comparison
#   * simple_encoder_comparison
#   * skip_encoder_comparison
# - Runs: 30 runs per experiment (6 lr × 5 bs)
#   * lr1e-05_bs8_epochs2 (for testing)
#   * lr1e-05_bs16_epochs2 (for testing)
#   * etc.
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
# export MLFLOW_EXPERIMENT_NAME=encoder_test_experiment
mkdir -p "$MLFLOW_TRACKING_URI"

export CUDA_VISIBLE_DEVICES=0

# ==========================================
# MODIFY THESE PARAMETERS AS NEEDED:
# ==========================================
EPOCHS=10000
LEARNING_RATE=1e-4
BATCH_SIZE=32

# Pass the array task ID to the Python script
/nfs/nhome/live/lporta/.conda/envs/decoders/bin/python neurodecoders/encoder/mlflow_training.py \
    --array-task-id $SLURM_ARRAY_TASK_ID \
    --epochs $EPOCHS \
    --mixed-precision \
    --early-stopping \
    --early-stopping-patience 100 \
    --checkpointing \
    --gradient-clip-val 1.0 \
    --n-neurons 100 \
    --n-images 10000 \
    --experiment-name "second_run" \
    --run-name "lr${LEARNING_RATE}_bs${BATCH_SIZE}_epochs${EPOCHS}_task${SLURM_ARRAY_TASK_ID}"

echo "Enhanced training completed for task $SLURM_ARRAY_TASK_ID"
echo "MLflow experiments should be available in: ${CURRENT_DIR}/mlruns"
