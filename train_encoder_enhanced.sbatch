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
#SBATCH -o /ceph/margrie/laura/neurodecoders/slurm/enhanced_%A_%a.out.txt # STDOUT
#SBATCH -e /ceph/margrie/laura/neurodecoders/slurm/enhanced_%A_%a.err.txt # STDERR

echo "=== Enhanced Encoder Training ==="
echo "Initialization completed."
echo "Activating the environment and setting up the package"

module load miniconda
conda activate /nfs/nhome/live/lporta/.conda/envs/decoders

echo "Setting up environment variables"
export CUDA_VISIBLE_DEVICES=0

echo "Starting enhanced array..."
echo "Array task id: $SLURM_ARRAY_TASK_ID"
echo "Node name: $SLURMD_NODENAME"

# ==========================================
# MODIFY THESE PARAMETERS AS NEEDED:
# ==========================================
EPOCHS=2  # Start with 2 epochs for testing, then increase to 10000 for full training
# For full training, change to: EPOCHS=10000

# Enhanced hyperparameter sweep with better defaults
# 3 model types x 6 learning rates x 5 batch sizes = 90 combinations
# Model types: [resnet, simple, skip]
# Learning rates: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
# Batch sizes: [8, 16, 32, 64, 128]

# Pass the array task ID to the Python script
python neurodecoders/encoder/mlflow_training.py \
    --array-task-id $SLURM_ARRAY_TASK_ID \
    --epochs $EPOCHS \
    --enable-mixed-precision \
    --enable-early-stopping \
    --early-stopping-patience 100 \
    --enable-checkpointing \
    --gradient-clip-val 1.0

echo "Enhanced training completed for task $SLURM_ARRAY_TASK_ID"
