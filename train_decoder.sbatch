#!/bin/bash
#
#SBATCH -p gpu
#SBATCH -J decoder_training
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 02:00:00
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

export CUDA_VISIBLE_DEVICES=0

echo "[INFO] Launching decoder training"
/nfs/nhome/live/lporta/.conda/envs/decoders/bin/python neurodecoders/decoder/decoder.py \
  --dataset "synthdata_dataset-mnist_sta-periodic_patterns,70,70_n_neurons-100_n_images-10000_datetime-20250827_121741.npz"

echo "[INFO] End time: $(date)"
