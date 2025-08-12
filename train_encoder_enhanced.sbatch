#!/bin/bash
#
#SBATCH -p gpu # partition (queue)
#SBATCH -J encoder_enhanced_training
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 00-06:00 # time (D-HH:MM)
#SBATCH --array=0-29%6 # 30 combinations (6 learning rates x 5 batch sizes), max 6 running simultaneously
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

# Enhanced hyperparameter sweep with better defaults
# 6 learning rates x 5 batch sizes = 30 combinations
# Learning rates: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
# Batch sizes: [8, 16, 32, 64, 128]

# Pass the array task ID to the Python script
python neurodecoders/encoder/mlflow_training.py \
    --array-task-id $SLURM_ARRAY_TASK_ID \
    --enable-mixed-precision \
    --enable-early-stopping \
    --early-stopping-patience 100 \
    --enable-checkpointing \
    --gradient-clip-val 1.0

echo "Enhanced training completed for task $SLURM_ARRAY_TASK_ID"
