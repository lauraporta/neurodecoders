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

MODEL_TYPE="resnet"           # Options: simple, resnet, skip
EPOCHS=10000                     # Number of training epochs
DATASET_TYPE="mnist"        # Dataset type
STA_TYPE="perlin_noise_patterns,11,11"  # STA type
N_NEURONS=1000               # Number of neurons
N_IMAGES=1000                # Number of images
EXPERIMENT_NAME="resnet_hyperparameter_sweep"  # MLflow experiment name

# Training parameters
OPTIMIZER="adam"             # Options: adam, adamw, sgd
LOSS_FUNCTION="mse"          # Options: mse, l1, smooth_l1, huber
WEIGHT_DECAY=0.0             # Weight decay (L2 regularization)
SCHEDULER="none"             # Options: none, step, cosine, plateau

# Define arrays for hyperparameter sweep
LEARNING_RATES=(0.0001 0.001 0.01)
BATCH_SIZES=(16 32 64)

# ==========================================
# TRAINING LOOP (DO NOT MODIFY BELOW):
# ==========================================

for lr in "${LEARNING_RATES[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
        # Create run name based on current parameters
        RUN_NAME="resnet_lr${lr}_bs${batch_size}"

        echo "=========================================="
        echo "Training with parameters:"
        echo "  Model Type: $MODEL_TYPE"
        echo "  Learning Rate: $lr"
        echo "  Batch Size: $batch_size"
        echo "  Run Name: $RUN_NAME"
        echo "=========================================="

        python neurodecoders/encoder/mlflow_training.py \
            --model-type $MODEL_TYPE \
            --epochs $EPOCHS \
            --dataset-type $DATASET_TYPE \
            --sta-type $STA_TYPE \
            --n-neurons $N_NEURONS \
            --n-images $N_IMAGES \
            --experiment-name $EXPERIMENT_NAME \
            --run-name $RUN_NAME \
            --batch-size $batch_size \
            --learning-rate $lr \
            --optimizer $OPTIMIZER \
            --loss-function $LOSS_FUNCTION \
            --weight-decay $WEIGHT_DECAY \
            --scheduler $SCHEDULER

        echo "Completed training for $RUN_NAME"
        echo "=========================================="
    done
done

echo "All encoder training runs completed."
