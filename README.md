# NeuroDecoders

Neural encoder training and synthetic dataset generation for neuroscience research.

## Quick Start

### 1. Create Synthetic Dataset

Generate synthetic neural responses with configurable parameters:

```bash
python neurodecoders/synthetic/create_simulated_neural_responses.py \
    --n_images 100 \
    --n_neurons 50 \
    --dataset_type cifar10 \
    --sta_type perlin_noise_patterns,11,11
```

**Arguments:**
- `--n_images`: Number of images (default: 100)
- `--n_neurons`: Number of neurons (default: 100)
- `--dataset_type`: Dataset type (default: cifar10)
- `--sta_type`: STA type and parameters (default: periodic_patterns,70,70)

### 2. Train Encoder with MLflow

Train neural encoders with experiment tracking:

```bash
python neurodecoders/encoder/mlflow_training.py \
    --model-type simple \
    --out-neurons 100 \
    --learning-rate 0.001 \
    --epochs 30 \
    --batch-size 32 \
    --dataset-type cifar10 \
    --sta-type perlin_noise_patterns,11,11 \
    --n-neurons 1000 \
    --n-images 1000 \
    --experiment-name my_experiment \
    --run-name simple_encoder_v1
```

**Arguments:**
- `--model-type`: Model type (simple, resnet)
- `--out-neurons`: Number of output neurons
- `--learning-rate`: Learning rate
- `--epochs`: Number of epochs
- `--batch-size`: Batch size
- `--dataset-type`: Dataset type (cifar10, mnist)
- `--sta-type`: STA type for synthetic data
- `--n-neurons`: Number of neurons in synthetic data
- `--n-images`: Number of images in synthetic data
- `--experiment-name`: MLflow experiment name
- `--run-name`: MLflow run name

### 3. View MLflow Experiments

```bash
mlflow ui
# Open http://localhost:5000 in your browser
```

## Advanced Usage

### Configuration Files

Use JSON configuration files for complex experiments:

```bash
python neurodecoders/encoder/mlflow_training.py \
    --mode config \
    --config neurodecoders/encoder/configs/single_experiment.json
```

### Experiment Comparison

Run multiple experiments for comparison:

```bash
python neurodecoders/encoder/mlflow_training.py \
    --mode comparison
```

### Hyperparameter Sweep

For SLURM job arrays:

```bash
python neurodecoders/encoder/mlflow_training.py \
    --mode hyperparameter_sweep \
    --array-task-id 0
```

## Output

- **Synthetic datasets**: `workspace/datasets/synthetic/`
- **Analysis plots**: `workspace/plots/analysis/`
- **Trained models**: `workspace/models/encoders/`
- **MLflow experiments**: `mlruns/`
