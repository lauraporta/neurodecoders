#!/bin/bash
#
#SBATCH -p gpu # partition (queue)
#SBATCH -J encoder_hyperparameter_sweep
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 00-06:00 # time (D-HH:MM)
#SBATCH --array=0-15%4 # 16 combinations (4 learning rates x 4 batch sizes), max 4 running simultaneously
#SBATCH -o /ceph/margrie/laura/neurodecoders/slurm/slurm_%A_%a.out.txt # STDOUT
#SBATCH -e /ceph/margrie/laura/neurodecoders/slurm/slurm_%A_%a.err.txt # STDERR

echo "Initialization completed."
echo "Activating the environment and setting up the package"

module load miniconda
conda activate /nfs/nhome/live/lporta/.conda/envs/decoders

echo "Setting up environment variables"
export CUDA_VISIBLE_DEVICES=0

echo "Starting array..."
echo "Array task id: $SLURM_ARRAY_TASK_ID"
echo "Node name: $SLURMD_NODENAME"

# Pass the array task ID to the Python script
python neurodecoders/encoder/mlflow_training.py \
    --mode hyperparameter_sweep \
    --array-task-id $SLURM_ARRAY_TASK_ID
