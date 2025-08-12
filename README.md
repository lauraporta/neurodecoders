# NeuroDecoders

Neural encoder training and synthetic dataset generation for neuroscience research.

## Configuration

The project uses a simple configuration system to manage artifact paths. See [CONFIGURATION.md](CONFIGURATION.md) for details.

**Quick setup**: Edit `config.yaml` to change where all data is stored:

```yaml
base_path: "."  # Change this to your desired directory
```

**Usage in code**:
```python
from neurodecoders.paths import get_path, get_synthetic_data_path
data_dir = get_synthetic_data_path()  # Gets configured path
```

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
    --learning-rate 0.001 \
    --epochs 30 \
    --batch-size 32 \
    --optimizer adam \
    --weight-decay 0.0 \
    --loss-function mse \
    --scheduler none \
    --dataset-type cifar10 \
    --sta-type periodic_patterns,70,70 \
    --n-neurons 100 \
    --n-images 100000 \
    --experiment-name my_experiment \
    --run-name simple_encoder_v1
```

**Model Arguments:**
- `--model-type`: Model type (simple, skip, resnet)
- `--resnet-type`: ResNet type (resnet18, resnet34, resnet50) - only for resnet model
- `--freeze-backbone`: Freeze ResNet backbone (default: True)
- `--unfreeze-backbone`: Unfreeze ResNet backbone (overrides --freeze-backbone)

**Training Arguments:**
- `--learning-rate`: Learning rate (default: 0.001)
- `--epochs`: Number of epochs (default: 30)
- `--batch-size`: Batch size (default: 32)
- `--optimizer`: Optimizer type (adam, adamw, sgd) (default: adam)
- `--weight-decay`: Weight decay/L2 regularization (default: 0.0)
- `--loss-function`: Loss function (mse, l1, smooth_l1, huber) (default: mse)
- `--scheduler`: Learning rate scheduler (none, step, cosine, plateau) (default: none)
- `--scheduler-step-size`: Step size for step scheduler (default: 30)
- `--scheduler-gamma`: Gamma for step scheduler (default: 0.1)

**Data Arguments:**
- `--dataset-type`: Dataset type (cifar10, mnist) (default: cifar10)
- `--sta-type`: STA type for synthetic data (default: periodic_patterns,70,70)
- `--n-neurons`: Number of neurons in synthetic data (default: 100)
- `--n-images`: Number of images in synthetic data (default: 100000)

**Data Loading Arguments:**
- `--use-memory-mapping`: Use memory mapping for large datasets
- `--chunk-size`: Chunk size for data loading (default: 10000)
- `--prefetch-factor`: DataLoader prefetch factor (default: 2)
- `--num-workers`: Number of workers for data loading (default: 0)
- `--pin-memory`: Pin memory for faster GPU transfer (default: True)
- `--no-pin-memory`: Disable pin memory (overrides --pin-memory)

**MLflow Arguments:**
- `--experiment-name`: MLflow experiment name (default: neural_encoder)
- `--run-name`: MLflow run name

**Note:** The `--out-neurons` parameter is now optional and will be automatically inferred from the dataset if not specified.

### 3. View MLflow Experiments

```bash
mlflow ui
# Open http://localhost:5000 in your browser
```

## Advanced Usage

### Hyperparameter Sweep

For SLURM job arrays:

```bash
python neurodecoders/encoder/mlflow_training.py \
    --array-task-id 0
```

### Large Dataset Training with Memory Optimization

For training with large datasets (100k+ images), use memory mapping and optimized data loading:

```bash
python neurodecoders/encoder/mlflow_training.py \
    --model-type simple \
    --learning-rate 0.001 \
    --epochs 100 \
    --batch-size 128 \
    --n-images 100000 \
    --n-neurons 100 \
    --use-memory-mapping \
    --num-workers 4 \
    --prefetch-factor 3 \
    --pin-memory \
    --experiment-name large_dataset_training \
    --run-name simple_large_optimized
```

**Memory Optimization Tips:**
- Use `--use-memory-mapping` for datasets > 1GB
- Increase `--num-workers` for faster data loading (4-8 workers recommended)
- Use `--pin-memory` for GPU training
- Adjust `--batch-size` based on available GPU memory

## Output

- **Synthetic datasets**: `workspace/datasets/synthetic/`
- **Analysis plots**: `workspace/plots/analysis/`
- **Trained models**: `workspace/models/encoders/`
- **MLflow experiments**: `mlruns/`
