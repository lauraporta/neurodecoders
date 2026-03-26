#!/bin/bash
#
# Quick refactor test - minimal epochs
#
#SBATCH -p gpu_lowp,a100
#SBATCH -J refactor_test
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 00-00:30
#SBATCH -o /ceph/margrie/laura/neurodecoders/slurm/refactor_test_%j_%a.out.txt
#SBATCH -e /ceph/margrie/laura/neurodecoders/slurm/refactor_test_%j_%a.err.txt
#SBATCH --exclude=gpu-sr670-21,gpu-sr670-22

echo "[INFO] Refactor Test - MODE=$MODE"
echo "[INFO] SLURM_JOB_ID=$SLURM_JOB_ID"
echo "[INFO] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Start time: $(date)"

# Load modules and activate conda
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
module load miniconda 2>/dev/null || true

eval "$(conda shell.bash hook)"
conda activate /nfs/nhome/live/lporta/.conda/envs/decoders
which python

export CUDA_VISIBLE_DEVICES=0

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "[INFO] MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"
fi

mkdir -p /ceph/margrie/laura/neurodecoders/workspace
mkdir -p /ceph/margrie/laura/neurodecoders/slurm

# ============================================================
# TEST PARAMETERS - minimal for quick validation
# ============================================================
ENCODER_RUN_ID_FILE="/ceph/margrie/laura/neurodecoders/workspace/encoder_run_id_refactor_test.txt"
EXPERIMENT_NAME="architecture_comparison_refactor_test"

# Use standard dataset (must exist)
N_NEURONS=5000
N_TRAIN_IMAGES=8000
N_TEST_IMAGES=2000
DATASET_TYPE="cifar10"
STA_TYPE="gabor,11,11"

# Very few epochs for quick test
TEST_EPOCHS=3

COMMON_ARGS="--n-neurons $N_NEURONS --n-train-images $N_TRAIN_IMAGES --n-test-images $N_TEST_IMAGES --dataset-type $DATASET_TYPE --sta-type $STA_TYPE --experiment-name $EXPERIMENT_NAME"

load_encoder_run_id() {
    if [ ! -f "$ENCODER_RUN_ID_FILE" ]; then
        echo "[ERROR] Encoder run_id file not found: $ENCODER_RUN_ID_FILE"
        echo "[ERROR] Please run with MODE=encoder first"
        exit 1
    fi
    ENCODER_RUN_ID=$(cat "$ENCODER_RUN_ID_FILE")
    echo "[INFO] Using encoder run_id: $ENCODER_RUN_ID"
}

run_encoder() {
    echo "[INFO] Training encoder (test mode: $TEST_EPOCHS epochs)..."
    
    python -m neurodecoders.experiments.architecture_comparison \
        --mode encoder \
        $COMMON_ARGS \
        --encoder-epochs $TEST_EPOCHS \
        --encoder-lr 0.001 \
        2>&1 | tee /tmp/encoder_output_${SLURM_JOB_ID}.txt
    
    ENCODER_RUN_ID=$(grep -oP "Encoder run_id: \K[a-f0-9]+" /tmp/encoder_output_${SLURM_JOB_ID}.txt | tail -1)
    
    if [ -n "$ENCODER_RUN_ID" ]; then
        echo "$ENCODER_RUN_ID" > "$ENCODER_RUN_ID_FILE"
        echo "[SUCCESS] Saved encoder run_id to $ENCODER_RUN_ID_FILE"
    else
        echo "[ERROR] Could not extract encoder run_id"
        exit 1
    fi
}

run_approaches() {
    APPROACHES=("input_optim" "diffusion" "guided")
    APPROACH=${APPROACHES[$SLURM_ARRAY_TASK_ID]}
    echo "[INFO] Running approach: $APPROACH (test mode)"
    
    case $APPROACH in
        "input_optim")
            load_encoder_run_id
            python -m neurodecoders.experiments.architecture_comparison \
                --mode input_optim \
                --encoder-run-id "$ENCODER_RUN_ID" \
                $COMMON_ARGS \
                --input-optim-steps 10 \
                --input-optim-lr 0.1 \
                --input-optim-loss poisson_mean \
                --input-optim-blur-sigma 1.5 \
                --input-optim-scheduler none
            ;;
        
        "diffusion")
            python -m neurodecoders.experiments.architecture_comparison \
                --mode diffusion \
                $COMMON_ARGS \
                --diffusion-epochs $TEST_EPOCHS \
                --diffusion-lr 0.001
            ;;
        
        "guided")
            load_encoder_run_id
            python -m neurodecoders.experiments.architecture_comparison \
                --mode guided \
                --encoder-run-id "$ENCODER_RUN_ID" \
                $COMMON_ARGS \
                --guided-epochs $TEST_EPOCHS \
                --guided-lr 0.0001 \
                --guided-tv-weight 0.01
            ;;
    esac
}

case ${MODE:-encoder} in
    "encoder")
        run_encoder
        ;;
    "approaches")
        run_approaches
        ;;
    *)
        echo "[ERROR] Unknown MODE: $MODE"
        exit 1
        ;;
esac

EXIT_CODE=$?
echo "[INFO] End time: $(date)"
echo "[INFO] Exit code: $EXIT_CODE"
exit $EXIT_CODE
