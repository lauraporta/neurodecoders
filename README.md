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

Generate synthetic neural responses matching paper's scale (32x32 images, ~100 neurons):

```bash
python neurodecoders/synthetic/create_simulated_neural_responses.py \
    --n_images 5000 \
    --n_neurons 1000 \
    --dataset_type cifar10 \
    --sta_type gabor,11,11 \
    --batch_size 100 \
    --neuron_batch_size 100
```

**Arguments:**
- `--n_images`: Number of images (default: 100, paper uses 4500+500)
- `--n_neurons`: Number of neurons (default: 100)
- `--dataset_type`: Dataset type (default: cifar10, gives 32x32 images)
- `--sta_type`: STA type and parameters (default: periodic_patterns,7,7)
  - **IMPORTANT:** For 32x32 images, use small kernels: 5x5, 7x7, or 11x11
  - Kernels should be ~20-30% of image size for realistic RFs
  - Examples: `gabor,5,5`, `gabor,7,7`, `periodic_patterns,11,11`
- `--batch_size`: Image batch size for memory-efficient processing (default: 100)
- `--neuron_batch_size`: Neuron batch size for memory-efficient processing (default: 1000)

**Memory Optimization Tips:**
- For large neuron counts (5000+), use smaller batch sizes: `--batch_size 50 --neuron_batch_size 500`
- Monitor memory usage: the script shows estimated memory before processing
- Target total memory < 4-6 GB to avoid out-of-memory kills
- Smaller batch sizes = slower but more memory-safe

### 2. Train Simple3Layer Encoder (Paper-like Architecture)

Train the new spatial readout encoder matching the paper's approach:

```bash
python neurodecoders/encoder/mlflow_training.py \
    --model-type simple3layer \
    --learning-rate 0.001 \
    --epochs 100 \
    --batch-size 32 \
    --optimizer adam \
    --weight-decay 0.0001 \
    --loss-function mse \
    --scheduler plateau \
    --dataset-type cifar10 \
    --sta-type gabor,7,7 \
    --n-neurons 100 \
    --n-images 5000 \
    --experiment-name simple3layer_spatial_readout \
    --run-name gabor_7x7_100n
```

**Key Model: Simple3LayerEncoder**
- 3 convolutional layers extracting features (matching paper)
- **Spatial readout layer** that learns RF positions for each neuron
- Independent linear readout per neuron from features at its RF
- This is the CRITICAL difference enabling reconstruction!

**Alternative: For comparison with old models**
```bash
# Old simple encoder (no spatial readout - won't reconstruct well)
python neurodecoders/encoder/mlflow_training.py \
    --model-type simple \
    --learning-rate 0.001 \
    --epochs 50 \
    --batch-size 32 \
    --dataset-type cifar10 \
    --sta-type gabor,7,7
```

**Model Arguments:**
- `--model-type`: Model type 
  - `simple3layer` - **RECOMMENDED**: 3-layer CNN with spatial readout (matches paper)
  - `simple` - Old simple encoder without spatial readout
  - `skip` - Simple encoder with skip connections
  - `resnet` - ResNet-based encoder (not recommended for reconstruction)
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
  - CIFAR-10 gives 32x32 images (close to paper's 36x64)
  - MNIST gives 28x28 images
- `--sta-type`: STA type for synthetic data (default: periodic_patterns,7,7)
  - **Recommended sizes for 32x32 images:** 5x5, 7x7, 11x11
  - Example patterns: `gabor,7,7`, `periodic_patterns,5,5`, `perlin_noise_patterns,11,11`
- `--n-neurons`: Number of neurons in synthetic data (default: 100)
  - Paper uses ~30 neurons for evaluation
  - 100-200 neurons is good for initial experiments
- `--n-images`: Number of images in synthetic data (default: 100000)
  - Paper uses 4500 for training + 500 for validation
  - Start with 5000-10000 for quick experiments

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

### 3. Reconstruct Images via Input Optimization

Use the trained encoder to reconstruct images from neural responses:

```bash
python neurodecoders/input_optim/mlflow_run.py \
    --encoder-run-id <your_mlflow_run_id> \
    --dataset-path workspace/datasets/synthetic/test/synthdata_*.npz \
    --n-images 10 \
    --image-size 32 \
    --steps 1000 \
    --learning-rate 0.05 \
    --loss mse \
    --experiment-name reconstruction_test \
    --run-name simple3layer_reconstruction
```

**Key Parameters:**
- `--encoder-run-id`: MLflow run ID of trained encoder (find in MLflow UI)
- `--dataset-path`: Path to test dataset with neural responses
- `--image-size`: Should match training (32 for CIFAR-10)
- `--steps`: Optimization steps (paper uses 1000, ~5 seconds)
- `--loss`: Loss function (`mse` recommended, matches paper)

**How it works (matching the paper):**
1. Start with blank/gray image
2. Forward pass through encoder → predicted responses
3. Compute MSE between predicted and target responses
4. Backward pass to get gradients w.r.t. input image
5. **Apply Gaussian blur (σ=2.5px) to gradients** (reduces high-frequency noise)
6. Update image via gradient descent
7. Repeat for 1000 steps

### 4. View MLflow Experiments

```bash
mlflow ui
# Open http://localhost:5000 in your browser
```

## Advanced Usage

### Complete Workflow Example

```bash
# 1. Generate synthetic data (5000 images, 100 neurons, 7x7 Gabor kernels)
python neurodecoders/synthetic/create_simulated_neural_responses.py \
    --n_images 5000 \
    --n_neurons 100 \
    --dataset_type cifar10 \
    --sta_type gabor,7,7

# 2. Train Simple3Layer encoder with spatial readout
python neurodecoders/encoder/mlflow_training.py \
    --model-type simple3layer \
    --learning-rate 0.001 \
    --epochs 100 \
    --batch-size 32 \
    --loss-function mse \
    --scheduler plateau \
    --sta-type gabor,7,7 \
    --n-neurons 100 \
    --experiment-name paper_approach

# 3. Get encoder run ID from MLflow UI, then reconstruct images
python neurodecoders/input_optim/mlflow_run.py \
    --encoder-run-id <run_id_from_step_2> \
    --dataset-path workspace/datasets/synthetic/test/synthdata_*.npz \
    --n-images 30 \
    --steps 1000 \
    --loss mse
```

### Hyperparameter Sweep

For SLURM job arrays:

```bash
python neurodecoders/encoder/mlflow_training.py \
    --array-task-id 0
```

### Large-Scale Training

For training with more neurons/images (closer to paper scale):

```bash
python neurodecoders/encoder/mlflow_training.py \
    --model-type simple3layer \
    --learning-rate 0.001 \
    --epochs 100 \
    --batch-size 64 \
    --n-images 10000 \
    --n-neurons 500 \
    --sta-type gabor,7,7 \
    --use-memory-mapping \
    --num-workers 4 \
    --pin-memory \
    --experiment-name large_scale_experiment
```

**Memory Optimization Tips:**
- Use `--use-memory-mapping` for datasets > 1GB
- Increase `--num-workers` for faster data loading (4-8 workers recommended)
- Use `--pin-memory` for GPU training
- Adjust `--batch-size` based on available GPU memory
- For synthetic data generation with many neurons, reduce batch sizes

## Output

- **Synthetic datasets**: `workspace/datasets/synthetic/`
- **Analysis plots**: `workspace/plots/analysis/`
- **Trained models**: `workspace/models/encoders/`
- **MLflow experiments**: `mlruns/`
