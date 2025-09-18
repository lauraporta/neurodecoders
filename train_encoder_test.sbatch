#!/bin/bash
#
#SBATCH -p gpu_lowp,gpu,a100 # partition (queue)
#SBATCH -J encoder_test
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 00:01:00
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
export MLFLOW_EXPERIMENT_NAME=encoder_test_experiment
mkdir -p "$MLFLOW_TRACKING_URI"

export CUDA_VISIBLE_DEVICES=0

echo "[INFO] Launching training with user-specified command"
/nfs/nhome/live/lporta/.conda/envs/decoders/bin/python neurodecoders/encoder/mlflow_training.py \
  --model-type simple \
  --epochs 100 \
  --dataset-type cifar10 \
  --sta-type gabor,100,100 \
  --n-neurons 100 \
  --n-images 8000 \
  --cv-folds 1 \
  --experiment-name with_plots \
  --run-name commit_$(git rev-parse --short HEAD)_test_run \

echo "[INFO] End time: $(date)"
