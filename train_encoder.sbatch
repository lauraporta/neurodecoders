#!/bin/bash
#
#SBATCH -p gpu # partition (queue)
#SBATCH -J encoder_training
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 00-12:00 # time (D-HH:MM)
#SBATCH -o /ceph/margrie/laura/neurodecoders/slurm/slurm.%N.%j.out.txt # STDOUT
#SBATCH -e /ceph/margrie/laura/neurodecoders/slurm/slurm.%N.%j.err.txt # STDERR

echo "Initialization completed."
echo "Activating the environment and setting up the package"

module load miniconda
conda activate /nfs/nhome/live/lporta/.conda/envs/decoders

echo "Setting up environment variables"
export CUDA_VISIBLE_DEVICES=0

echo "Starting encoder training..."

# ==========================================
# MODIFY THESE PARAMETERS AS NEEDED:
# ==========================================

MODEL_TYPE="resnet"           # Options: simple, resnet, skip_connection
RUN_NAME="resnet"             # Name for this training run
BATCH_SIZE=16                 # Adjust based on GPU memory (8, 16, 32)
EPOCHS=50                     # Number of training epochs
OUT_NEURONS=1000             # Number of output neurons
DATASET_TYPE="cifar10"        # Dataset type
STA_TYPE="perlin_noise_patterns,11,11"  # STA type
N_NEURONS=1000               # Number of neurons
N_IMAGES=1000                # Number of images
EXPERIMENT_NAME="compare_encoders"  # MLflow experiment name

# ==========================================
# TRAINING COMMAND (DO NOT MODIFY BELOW):
# ==========================================

python neurodecoders/encoder/mlflow_training.py \
    --mode single \
    --model-type $MODEL_TYPE \
    --out-neurons $OUT_NEURONS \
    --epochs $EPOCHS \
    --dataset-type $DATASET_TYPE \
    --sta-type $STA_TYPE \
    --n-neurons $N_NEURONS \
    --n-images $N_IMAGES \
    --experiment-name $EXPERIMENT_NAME \
    --run-name $RUN_NAME \
    --batch-size $BATCH_SIZE

echo "Encoder training completed."
