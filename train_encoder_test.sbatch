#!/bin/bash
# 
#SBATCH -p gpu
#SBATCH -J encoder_test
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 00:10:00
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

/nfs/nhome/live/lporta/.conda/envs/decoders/bin/python - <<'EOF'
import os, mlflow
print('[DEBUG] MLFLOW_TRACKING_URI =', os.environ.get('MLFLOW_TRACKING_URI'))
print('[DEBUG] MLFLOW_EXPERIMENT_NAME =', os.environ.get('MLFLOW_EXPERIMENT_NAME'))
try:
    exp = mlflow.get_experiment_by_name(os.environ.get('MLFLOW_EXPERIMENT_NAME',''))
    print('[DEBUG] Experiment exists:', bool(exp))
except Exception as e:
    print('[WARN] Experiment lookup failed:', e)
EOF

echo "[INFO] Launching training with user-specified command"
/nfs/nhome/live/lporta/.conda/envs/decoders/bin/python neurodecoders/encoder/mlflow_training.py \
    --model-type resnet \
    --epochs 3 \
    --dataset-type cifar10 \
    --sta-type periodic_patterns,70,70 \
    --n-neurons 10 \
    --n-images 10 \
    --experiment-name with_history \
    --run-name test1

status=$?
if [ $status -ne 0 ]; then
  echo "[ERROR] Training failed with status $status" >&2
  exit $status
fi

echo "[INFO] Training completed. Listing recent MLflow files (depth 3)."
if [ -d "$MLFLOW_TRACKING_URI" ]; then
  find "$MLFLOW_TRACKING_URI" -maxdepth 3 -type f -printf "%P\n" | head -n 50
fi

echo "[INFO] End time: $(date)"
