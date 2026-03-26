# NeuroDecoders

Neural encoder training and image reconstruction from synthetic neural responses.

The project compares four reconstruction approaches:

| Approach | Description |
|---|---|
| **Input optimisation** | Gradient-based image optimisation to match target firing rates |
| **Pure decoder** | MLP trained directly to map firing rates → images |
| **Diffusion decoder** | DDPM/DDIM diffusion model conditioned on neural activity |
| **Encoder-guided decoder** | Encoder-in-the-loop training with optional pixel-level auxiliary loss |

## Setup

Edit `config.yaml` to set where data and artifacts are stored:

```yaml
base_path: "/path/to/your/workspace"
```

Set the MLflow tracking URI in a `.env` file:

```
MLFLOW_TRACKING_URI=http://localhost:5000
```

## Workflow

### 1. Generate synthetic data

```bash
python neurodecoders/synthetic/create_simulated_neural_responses.py \
    --n_images 5000 \
    --n_neurons 100 \
    --dataset_type cifar10 \
    --sta_type gabor,7,7
```

For large neuron counts, reduce batch sizes to manage memory:
```bash
    --batch_size 50 --neuron_batch_size 500
```

### 2. Train encoder

```bash
python neurodecoders/encoder/mlflow_training.py \
    --model-type simple3layer \
    --learning-rate 0.001 \
    --epochs 100 \
    --batch-size 32 \
    --loss-function mse \
    --scheduler plateau \
    --sta-type gabor,7,7 \
    --n-neurons 100 \
    --experiment-name my_encoder
```

The `simple3layer` model (3-layer CNN with spatial readout) is recommended — it is the only architecture that enables effective reconstruction.

### 3. Reconstruct images

**Input optimisation** — gradient descent on the image pixel space:

```bash
python neurodecoders/input_optim/mlflow_run.py \
    --encoder-run-id <mlflow_run_id> \
    --dataset-path workspace/datasets/synthetic/test/synthdata_*.npz \
    --n-images 10 \
    --steps 1000 \
    --loss poisson_mean
```

**Decoder training** (diffusion / encoder-guided / pure decoder) — run the full architecture comparison:

```bash
python neurodecoders/experiments/architecture_comparison.py \
    --experiment-name my_comparison \
    --encoder-run-id <mlflow_run_id>
```

Results are logged to MLflow. Visualise them with:

```bash
python neurodecoders/experiments/plot_architecture_comparison.py \
    --experiment-name my_comparison
```

## Demo

See [notebooks/encode_fellowship_demo.ipynb](notebooks/encode_fellowship_demo.ipynb) for an end-to-end walkthrough comparing all four reconstruction approaches.

## Cluster (SLURM)

Scripts for running experiments on a SLURM cluster are in [`sbatch_scripts/`](sbatch_scripts/):

| Script | Purpose |
|---|---|
| `create_synthetic_data.sbatch` | Generate synthetic dataset |
| `train_encoder_enhanced.sbatch` | Train encoder |
| `train_input_optim.sbatch` | Input optimisation sweep |
| `train_decoder.sbatch` | Train pure/diffusion decoder |
| `train_encoder_guided.sbatch` | Train encoder-guided decoder |
| `architecture_comparison.sbatch` | Full multi-mode comparison (encoder → approaches → analysis) |
| `guided_sweep.sbatch` / `guided_sweep_v2.sbatch` | Hyperparameter sweep for guided decoder |

Example (architecture comparison, all modes):

```bash
# 1. Train encoder
sbatch --export=MODE=encoder sbatch_scripts/architecture_comparison.sbatch

# 2. Run all reconstruction approaches (after encoder completes)
sbatch --array=0-4 --dependency=afterok:<job_id> \
    --export=MODE=approaches sbatch_scripts/architecture_comparison.sbatch
```

## Outputs

| Path | Contents |
|---|---|
| `workspace/datasets/synthetic/` | Synthetic neural response datasets |
| `workspace/models/` | Saved model checkpoints |
| `mlruns/` | MLflow experiment logs |
