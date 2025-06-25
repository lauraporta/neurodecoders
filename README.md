# neurodecoders

## GPUs available on the cluster

| GPU Model     | Nodes  | GPUs per Node      | Total GPUs  | RAM (GB) | FP32 TFLOPs | Tensor TFLOPs (FP16)       | FP16 Support |
|---------------|--------|--------------------|-------------|----------|-------------|----------------------------|--------------|
| **A4500**     | 5      | 3                  | 15          | 20       | ~23.7       | ~312 (with sparsity)       | ✅ Yes        |
| **P5000**     | 3      | 2, 1, 1            | 4           | 16       | ~8.9        | ❌                         | ❌ No         |
| **RTX 4000**  | 1      | 1                  | 1           | 8        | ~7.1        | ~57                        | ✅ Yes        |
| **RTX 2080**  | 2      | 1, 1               | 2           | 8        | ~10.1       | ~113                       | ✅ Yes        |
| **RTX 5000**  | 4      | 2                  | 8           | 16       | ~11.2       | ~130                       | ✅ Yes        |
| **A100**      | 4      | 4                  | 16          | 40       | ~19.5       | Up to 312 (TF32/FP16)      | ✅ Yes        |
| **L40S**      | 3      | 8                  | 24          | 48       | ~91.6       | ~1450 (with sparsity)      | ✅ Yes        |
| **H100**      | 2      | 4, 8               | 12          | 80       | ~30         | Up to 989 (with sparsity)  | ✅ Yes        |

### GPU recommendations

| Scenario                  | Model/Dataset Fit | Technique               | Recommended GPUs | Preferred Node Setup       |
|---------------------------|-------------------|--------------------------|------------------|-----------------------------|
| Small model + data        | ✅                | Single-GPU              | A4500, A100      | Any                         |
| Large batch               | ✅                | DataParallel            | A100, L40S       | Multi-GPU on 1 node         |
| Model too large           | ❌                | ZeRO / FSDP             | A100, H100       | Multi-GPU, fast interconnect|
| Transformers (GPT-like)   | ❌                | Tensor / Pipeline Parallelism | H100, L40S | Multi-node, NVLink/NVSwitch |

## GPU Memory Analysis

Use `analyze_model_gpu_needs.py` to estimate GPU memory requirements for your models:

```bash
# Basic usage
python analyze_model_gpu_needs.py --num-images 10000 --num-neurons 500

# Custom configuration
python analyze_model_gpu_needs.py \
    --batch-size 64 \
    --image-size 128 \
    --out-neurons 200 \
    --num-images 10000 \
    --num-neurons 500 \
    --dtype float16
```

**Required arguments:**
- `--num-images`: Number of images in dataset
- `--num-neurons`: Number of neurons in dataset

**Optional arguments:**
- `--batch-size`: Training batch size (default: 32)
- `--image-size`: Image dimensions (default: 128)
- `--out-neurons`: Model output neurons (default: 200)
- `--dtype`: Data type (float32/float16, default: float32)
- `--models`: Models to analyze (default: SimpleEncoder, SimpleDecoder)

The script provides:
- Memory breakdown (parameters, gradients, optimizer, activations)
- GPU recommendations based on cluster specs
- SLURM command generation
- Optimization suggestions
