#!/bin/bash
#
# ============================================================
# ARCHITECTURE COMPARISON EXPERIMENT (UNIFIED)
# ============================================================
# Single script for all architecture comparison experiments.
#
# MODES (set via --export=MODE=<mode>):
#   encoder         - Train encoder (single job, ~6 hours)
#   approaches      - Run 4 reconstruction approaches (array 0-3, ~12 hours)
#   ddim_steps      - Compare DDIM at different step counts (array 0-4, ~2 hours)
#   tv_sweep        - TV weight sweep for guided decoder (array 0-5, ~4 hours)
#   improvements    - Run improved decoder variants (array 0-1, ~6 hours)
#
# USAGE EXAMPLES:
#   # Train encoder first
#   sbatch --export=MODE=encoder architecture_comparison.sbatch
#
#   # Run all approaches (after encoder completes)
#   sbatch --array=0-4 --export=MODE=approaches architecture_comparison.sbatch
#
#   # With dependency on encoder job
#   sbatch --array=0-4 --dependency=afterok:12345 --export=MODE=approaches architecture_comparison.sbatch
#
#   # Single approach only
#   sbatch --array=0 --export=MODE=approaches architecture_comparison.sbatch   # input_optim
#   sbatch --array=1 --export=MODE=approaches architecture_comparison.sbatch   # diffusion (DDPM)
#   sbatch --array=2 --export=MODE=approaches architecture_comparison.sbatch   # diffusion_ddim
#   sbatch --array=3 --export=MODE=approaches architecture_comparison.sbatch   # guided
#   sbatch --array=4 --export=MODE=approaches architecture_comparison.sbatch   # pure_decoder
#
#   # DDIM step count comparison (requires trained model - runs inference only)
#   sbatch --array=0-4 --export=MODE=ddim_steps architecture_comparison.sbatch
#
#   # TV weight sweep
#   sbatch --array=0-5 --export=MODE=tv_sweep architecture_comparison.sbatch
#
#   # Improved architectures
#   sbatch --array=0-1 --export=MODE=improvements architecture_comparison.sbatch
#
# OUTPUT:
#   Encoder run_id saved to: /ceph/margrie/laura/neurodecoders/workspace/encoder_run_id.txt
# ============================================================
#
#SBATCH -p gpu_lowp,a100
#SBATCH -J arch_comp
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 00-12:00
#SBATCH -o /ceph/margrie/laura/neurodecoders/slurm/arch_%j_%a.out.txt
#SBATCH -e /ceph/margrie/laura/neurodecoders/slurm/arch_%j_%a.err.txt
#SBATCH --exclude=gpu-sr670-21,gpu-sr670-22

# ============================================================
# ENVIRONMENT SETUP (common to all modes)
# ============================================================

echo "[INFO] Architecture Comparison - MODE=$MODE"
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

# Create output directories
mkdir -p /ceph/margrie/laura/neurodecoders/workspace
mkdir -p /ceph/margrie/laura/neurodecoders/slurm

# ============================================================
# COMMON PARAMETERS
# ============================================================
ENCODER_RUN_ID_FILE="/ceph/margrie/laura/neurodecoders/workspace/encoder_run_id.txt"
EXPERIMENT_NAME="input_optim_debug"

# Dataset configuration (shared across all modes)
N_NEURONS=5000
N_TRAIN_IMAGES=8000
N_TEST_IMAGES=2000        # Revert to full test set size (must match generated data)
N_RECONSTRUCT_IMAGES=5    # Only reconstruct 5 images for speed

DATASET_TYPE="cifar10"
STA_TYPE="gabor,11,11"

# Common args for all python calls
COMMON_ARGS="--n-neurons $N_NEURONS --n-train-images $N_TRAIN_IMAGES --n-test-images $N_TEST_IMAGES --dataset-type $DATASET_TYPE --sta-type $STA_TYPE --experiment-name $EXPERIMENT_NAME --n-reconstruct-images $N_RECONSTRUCT_IMAGES"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

load_encoder_run_id() {
    if [ ! -f "$ENCODER_RUN_ID_FILE" ]; then
        echo "[ERROR] Encoder run_id file not found: $ENCODER_RUN_ID_FILE"
        echo "[ERROR] Please run with MODE=encoder first"
        exit 1
    fi
    ENCODER_RUN_ID=$(cat "$ENCODER_RUN_ID_FILE")
    echo "[INFO] Using encoder run_id: $ENCODER_RUN_ID"
}

# ============================================================
# MODE: encoder
# ============================================================

run_encoder() {
    echo "[INFO] Training encoder..."
    
    python -m neurodecoders.experiments.architecture_comparison \
        --mode encoder \
        $COMMON_ARGS \
        --encoder-epochs 100 \
        --encoder-lr 0.001 \
        2>&1 | tee /tmp/encoder_output_${SLURM_JOB_ID}.txt
    
    # Extract and save encoder run_id
    ENCODER_RUN_ID=$(grep -oP "Encoder run_id: \K[a-f0-9]+" /tmp/encoder_output_${SLURM_JOB_ID}.txt | tail -1)
    
    if [ -n "$ENCODER_RUN_ID" ]; then
        echo "$ENCODER_RUN_ID" > "$ENCODER_RUN_ID_FILE"
        echo "[SUCCESS] Saved encoder run_id to $ENCODER_RUN_ID_FILE"
    else
        echo "[ERROR] Could not extract encoder run_id"
        exit 1
    fi
}

# ============================================================
# MODE: approaches (array 0-3)
# ============================================================

run_approaches() {
    APPROACHES=("input_optim" "diffusion" "diffusion_ddim" "guided" "pure_decoder")
    APPROACH=${APPROACHES[$SLURM_ARRAY_TASK_ID]}
    echo "[INFO] Running approach: $APPROACH"
    
    case $APPROACH in
        "input_optim")
            load_encoder_run_id
            python -m neurodecoders.experiments.architecture_comparison \
                --mode input_optim \
                --encoder-run-id "$ENCODER_RUN_ID" \
                $COMMON_ARGS \
                --input-optim-steps 1000 \
                --input-optim-lr 0.1 \
                --input-optim-loss poisson_mean \
                --input-optim-blur-sigma 1.5 \
                --input-optim-scheduler none
            ;;
        
        "diffusion")
            # Standard DDPM with 1000 steps (stochastic)
            python -m neurodecoders.experiments.architecture_comparison \
                --mode diffusion \
                $COMMON_ARGS \
                --diffusion-epochs 100 \
                --diffusion-lr 0.001 \
                --diffusion-optimizer adam \
                --diffusion-sampler ddpm \
                --diffusion-inference-steps 1000
            ;;
        
        "diffusion_ddim")
            # DDIM with 50 steps (deterministic, ~20x faster inference)
            python -m neurodecoders.experiments.architecture_comparison \
                --mode diffusion_ddim \
                $COMMON_ARGS \
                --diffusion-epochs 100 \
                --diffusion-lr 0.001 \
                --diffusion-optimizer adam \
                --diffusion-sampler ddim \
                --diffusion-inference-steps 50 \
                --diffusion-eta 0.0
            ;;
        
        "guided")
            load_encoder_run_id
            python -m neurodecoders.experiments.architecture_comparison \
                --mode guided \
                --encoder-run-id "$ENCODER_RUN_ID" \
                $COMMON_ARGS \
                --guided-epochs 150 \
                --guided-lr 0.0001 \
                --guided-tv-weight 0.01
            ;;

        "pure_decoder")
            # Pure Decoder (Baseline)
            python -m neurodecoders.experiments.architecture_comparison \
                --mode pure_decoder \
                $COMMON_ARGS \
                --pure-decoder-epochs 100 \
                --pure-decoder-lr 0.001 \
                --pure-decoder-type simple \
                --pure-decoder-loss-fn mse
            ;;
    esac
}

# ============================================================
# MODE: tv_sweep (array 0-5)
# ============================================================

run_tv_sweep() {
    TV_WEIGHTS=(0.0 0.001 0.005 0.01 0.05 0.1)
    TV_WEIGHT=${TV_WEIGHTS[$SLURM_ARRAY_TASK_ID]}
    echo "[INFO] TV weight sweep: $TV_WEIGHT"
    
    load_encoder_run_id
    
    python -m neurodecoders.experiments.architecture_comparison \
        --mode guided \
        --encoder-run-id "$ENCODER_RUN_ID" \
        $COMMON_ARGS \
        --guided-epochs 150 \
        --guided-lr 0.0001 \
        --guided-tv-weight "$TV_WEIGHT"
}

# ============================================================
# MODE: ddim_steps (array 0-4)
# Compare DDIM quality at different step counts
# ============================================================

run_ddim_steps() {
    DDIM_STEPS=(10 20 50 100 200)
    STEPS=${DDIM_STEPS[$SLURM_ARRAY_TASK_ID]}
    echo "[INFO] DDIM step comparison: $STEPS steps"
    
    python -m neurodecoders.experiments.architecture_comparison \
        --mode diffusion_ddim \
        $COMMON_ARGS \
        --diffusion-epochs 100 \
        --diffusion-lr 0.001 \
        --diffusion-optimizer adam \
        --diffusion-sampler ddim \
        --diffusion-inference-steps "$STEPS" \
        --diffusion-eta 0.0
}

# ============================================================
# MODE: improvements (array 0-1)
# ============================================================

run_improvements() {
    EXPERIMENTS=("diffusion_deep" "guided_cnn")
    EXPERIMENT=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}
    echo "[INFO] Running improvement: $EXPERIMENT"
    
    case $EXPERIMENT in
        "diffusion_deep")
            python -m neurodecoders.experiments.architecture_comparison \
                --mode diffusion \
                $COMMON_ARGS \
                --diffusion-epochs 100 \
                --diffusion-lr 0.001 \
                --diffusion-optimizer adam \
                --diffusion-embed-type deep
            ;;
        
        "guided_cnn")
            load_encoder_run_id
            python -m neurodecoders.experiments.architecture_comparison \
                --mode guided \
                --encoder-run-id "$ENCODER_RUN_ID" \
                $COMMON_ARGS \
                --guided-epochs 150 \
                --guided-lr 0.0001 \
                --guided-decoder-type simple \
                --guided-tv-weight 0.01
            ;;
    esac
}

# ============================================================
# MAIN DISPATCH
# ============================================================

case ${MODE:-encoder} in
    "encoder")
        run_encoder
        ;;
    "approaches")
        run_approaches
        ;;
    "tv_sweep")
        run_tv_sweep
        ;;
    "ddim_steps")
        run_ddim_steps
        ;;
    "improvements")
        run_improvements
        ;;
    *)
        echo "[ERROR] Unknown MODE: $MODE"
        echo "[ERROR] Valid modes: encoder, approaches, tv_sweep, ddim_steps, improvements"
        exit 1
        ;;
esac

EXIT_CODE=$?

echo "[INFO] End time: $(date)"
echo "[INFO] Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] Completed successfully"
else
    echo "[ERROR] Failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
