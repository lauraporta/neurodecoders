#!/bin/bash
#
#SBATCH -p gpu_lowp,gpu # partition (queue)
#SBATCH -J create_synthetic_data
#SBATCH --mem=32G # Increased memory for large datasets
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 00-04:00 # time (D-HH:MM) - 4 hours should be enough
#SBATCH -o /ceph/margrie/laura/neurodecoders/slurm/synthetic_data_%j.out.txt
#SBATCH -e /ceph/margrie/laura/neurodecoders/slurm/synthetic_data_%j.err.txt

echo "[INFO] Creating synthetic neural response dataset"
echo "[INFO] SLURM_JOB_ID=$SLURM_JOB_ID"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Start time: $(date)"

# Load modules and activate conda environment
source /etc/profile.d/modules.sh
module load miniconda
conda activate /nfs/nhome/live/lporta/.conda/envs/decoders
which python

# Check GPU availability
nvidia-smi

# Run the synthetic data generation script
/nfs/nhome/live/lporta/.conda/envs/decoders/bin/python neurodecoders/synthetic/create_simulated_neural_responses.py \
    --n_images 10000 \
    --n_neurons 5000 \
    --dataset_type cifar10 \
    --sta_type gabor,11,11 \
    --batch_size 100 \
    --neuron_batch_size 100

echo "[INFO] Synthetic data creation completed"
echo "[INFO] End time: $(date)"
