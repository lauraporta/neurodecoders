#!/usr/bin/env python3
"""
GPU Memory Analysis Script for Neurodecoders

This script analyzes GPU memory requirements for PyTorch models including:
- Model memory estimation (parameters, gradients, optimizer)
- Dataset size estimation from data/ folder
- GPU recommendations based on cluster specs
- Memory optimization suggestions
"""

import argparse
import os
import glob
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

# Import models from local codebase
try:
    from neurodecoders.encoder.encoder import SimpleEncoder
    from neurodecoders.decoder.decoder import SimpleDecoder
except ImportError:
    print("Warning: Could not import models from neurodecoders package.")
    print("Make sure you're running this script from the repo root.")
    sys.exit(1)

# Try to import torchinfo for detailed model analysis
try:
    import torchinfo
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False
    print("Note: torchinfo not available. Install with: pip install torchinfo")


class GPUMemoryAnalyzer:
    """Analyzes GPU memory requirements for PyTorch models"""
    
    def __init__(self, dtype: str = "float32"):
        self.dtype = getattr(torch, dtype)
        self.dtype_size = self.dtype.itemsize
        self.gpu_specs = self._load_gpu_specs()
        
    def _load_gpu_specs(self) -> Dict:
        """Load GPU specifications from README.md"""
        gpu_specs = {
            "A4500": {"nodes": 5, "gpus_per_node": 3, "total_gpus": 15, "vram_gb": 20, "fp32_tflops": 23.7, "fp16_support": True},
            "P5000": {"nodes": 3, "gpus_per_node": [2, 1, 1], "total_gpus": 4, "vram_gb": 16, "fp32_tflops": 8.9, "fp16_support": False},
            "RTX 4000": {"nodes": 1, "gpus_per_node": 1, "total_gpus": 1, "vram_gb": 8, "fp32_tflops": 7.1, "fp16_support": True},
            "RTX 2080": {"nodes": 2, "gpus_per_node": [1, 1], "total_gpus": 2, "vram_gb": 8, "fp32_tflops": 10.1, "fp16_support": True},
            "RTX 5000": {"nodes": 4, "gpus_per_node": 2, "total_gpus": 8, "vram_gb": 16, "fp32_tflops": 11.2, "fp16_support": True},
            "A100": {"nodes": 4, "gpus_per_node": 4, "total_gpus": 16, "vram_gb": 40, "fp32_tflops": 19.5, "fp16_support": True},
            "L40S": {"nodes": 3, "gpus_per_node": 8, "total_gpus": 24, "vram_gb": 48, "fp32_tflops": 91.6, "fp16_support": True},
            "H100": {"nodes": 2, "gpus_per_node": [4, 8], "total_gpus": 12, "vram_gb": 80, "fp32_tflops": 30, "fp16_support": True}
        }
        return gpu_specs
    
    def estimate_model_memory(self, model: nn.Module, batch_size: int, 
                            input_size: Union[int, Tuple[int, int]], 
                            out_neurons: int = None) -> Dict:
        """Estimate memory usage for a PyTorch model"""
        
        # Calculate input dimensions
        if isinstance(input_size, int):
            input_shape = (batch_size, 1, input_size, input_size)
        else:
            input_shape = (batch_size, 1, input_size[0], input_size[1])
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Memory calculations (assuming float32 = 4 bytes)
        param_memory = total_params * 4  # bytes
        gradient_memory = trainable_params * 4  # bytes
        optimizer_memory = trainable_params * 8  # Adam optimizer uses 2x param size
        
        # Batch memory
        batch_memory = batch_size * input_shape[2] * input_shape[3] * self.dtype_size
        
        # Rough activation memory estimate (very approximate)
        # This is a simplified estimate - in practice it varies greatly
        activation_memory = batch_size * total_params * 0.1 * self.dtype_size  # Rough estimate
        
        # Try to get more accurate activation memory with torchinfo
        if TORCHINFO_AVAILABLE:
            try:
                info = torchinfo.summary(model, input_size=input_shape, verbose=0, device="cpu")
                activation_memory = info.total_mult_adds * self.dtype_size if info.total_mult_adds else activation_memory
            except:
                pass
        
        total_memory = param_memory + gradient_memory + optimizer_memory + batch_memory + activation_memory
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "param_memory_mb": param_memory / (1024 * 1024),
            "gradient_memory_mb": gradient_memory / (1024 * 1024),
            "optimizer_memory_mb": optimizer_memory / (1024 * 1024),
            "batch_memory_mb": batch_memory / (1024 * 1024),
            "activation_memory_mb": activation_memory / (1024 * 1024),
            "total_memory_mb": total_memory / (1024 * 1024),
            "input_shape": input_shape
        }
    
    def find_latest_dataset(self, data_dir: str = "data", num_images: Optional[int] = None) -> Optional[Dict]:
        """Find the most recently modified dataset file"""
        data_extensions = ['.npy', '.npz', '.pt', '.tif', '.h5', '.csv']
        data_files = []
        
        # Recursively search for data files
        for ext in data_extensions:
            pattern = os.path.join(data_dir, f"**/*{ext}")
            data_files.extend(glob.glob(pattern, recursive=True))
        
        if not data_files:
            return None
        
        # Find most recently modified file
        latest_file = max(data_files, key=os.path.getmtime)
        
        # Try to load and analyze the file
        try:
            file_info = self._analyze_data_file(latest_file)
            file_info["file_path"] = latest_file
            file_info["file_size_mb"] = os.path.getsize(latest_file) / (1024 * 1024)
            file_info["modified_time"] = datetime.fromtimestamp(os.path.getmtime(latest_file))
            
            # Override number of samples if specified
            if num_images is not None:
                file_info["num_samples"] = num_images
                # Recalculate total memory based on new sample count
                file_info["total_memory_mb"] = (file_info["per_sample_memory_bytes"] * num_images) / (1024 * 1024)
                file_info["user_specified_samples"] = True
            
            return file_info
        except Exception as e:
            print(f"Warning: Could not analyze {latest_file}: {e}")
            return {
                "file_path": latest_file,
                "file_size_mb": os.path.getsize(latest_file) / (1024 * 1024),
                "modified_time": datetime.fromtimestamp(os.path.getmtime(latest_file)),
                "error": str(e)
            }
    
    def _analyze_data_file(self, file_path: str) -> Dict:
        """Analyze a data file to estimate memory requirements"""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.npz':
            data = np.load(file_path)
            # Try to find the main data array
            if 'images' in data:
                array = data['images']
            elif 'responses' in data:
                array = data['responses']
            else:
                # Use the first array
                array = data[data.files[0]]
            data.close()
            
        elif ext == '.npy':
            array = np.load(file_path)
            
        elif ext == '.pt':
            data = torch.load(file_path, map_location='cpu')
            if isinstance(data, dict):
                # Try to find the main data tensor
                if 'images' in data:
                    array = data['images'].numpy()
                elif 'responses' in data:
                    array = data['responses'].numpy()
                else:
                    array = list(data.values())[0].numpy()
            else:
                array = data.numpy()
                
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Calculate memory estimates
        num_samples = array.shape[0] if array.ndim > 0 else 1
        per_sample_memory = array.nbytes / num_samples if num_samples > 0 else array.nbytes
        
        return {
            "num_samples": num_samples,
            "shape": array.shape,
            "dtype": str(array.dtype),
            "per_sample_memory_bytes": per_sample_memory,
            "total_memory_mb": array.nbytes / (1024 * 1024)
        }
    
    def recommend_gpu(self, total_memory_mb: float, batch_size: int, 
                     use_mixed_precision: bool = False) -> Dict:
        """Recommend GPU configuration based on memory requirements"""
        
        # Apply mixed precision savings if requested
        memory_factor = 0.5 if use_mixed_precision else 1.0
        adjusted_memory_mb = total_memory_mb * memory_factor
        
        # Add 20% buffer for safety
        required_memory_mb = adjusted_memory_mb * 1.2
        
        # Find suitable GPUs
        suitable_gpus = []
        for gpu_name, specs in self.gpu_specs.items():
            if specs["vram_gb"] * 1024 >= required_memory_mb:
                suitable_gpus.append((gpu_name, specs))
        
        if not suitable_gpus:
            return {
                "fits_on_single_gpu": False,
                "recommendation": "Model too large for available GPUs",
                "suggestions": [
                    "Use model parallelism across multiple GPUs",
                    "Reduce batch size",
                    "Use gradient checkpointing",
                    "Consider mixed precision training"
                ]
            }
        
        # Sort by VRAM (prefer smaller GPUs if they fit)
        suitable_gpus.sort(key=lambda x: x[1]["vram_gb"])
        best_gpu, best_specs = suitable_gpus[0]
        
        # Determine if we need multiple GPUs
        if best_specs["vram_gb"] * 1024 >= required_memory_mb * 2:  # If we have 2x memory
            gpus_needed = 1
            parallelism = "Single GPU"
        else:
            gpus_needed = max(1, int(np.ceil(required_memory_mb / (best_specs["vram_gb"] * 1024))))
            # Handle different gpus_per_node formats
            gpus_per_node = best_specs.get("gpus_per_node", 1)
            if isinstance(gpus_per_node, list):
                max_gpus_per_node = max(gpus_per_node)
            else:
                max_gpus_per_node = gpus_per_node
            parallelism = "Data Parallel" if gpus_needed <= max_gpus_per_node else "Multi-node"
        
        return {
            "fits_on_single_gpu": gpus_needed == 1,
            "recommended_gpu": best_gpu,
            "gpus_needed": gpus_needed,
            "parallelism_strategy": parallelism,
            "required_memory_mb": required_memory_mb,
            "gpu_vram_gb": best_specs["vram_gb"],
            "memory_utilization_percent": (required_memory_mb / (best_specs["vram_gb"] * 1024)) * 100,
            "suggestions": self._generate_suggestions(required_memory_mb, best_specs["vram_gb"] * 1024)
        }
    
    def _generate_suggestions(self, required_memory: float, available_memory: float) -> List[str]:
        """Generate optimization suggestions"""
        suggestions = []
        utilization = required_memory / available_memory
        
        if utilization > 0.8:
            suggestions.append("⚠️  High memory utilization - consider reducing batch size")
            suggestions.append("💡 Enable mixed precision training (float16)")
            suggestions.append("💡 Use gradient checkpointing to save memory")
        
        if utilization > 0.6:
            suggestions.append("💡 Consider using torch.compile() for optimization")
        
        if utilization < 0.3:
            suggestions.append("💡 You can increase batch size for better efficiency")
        
        return suggestions
    
    def generate_slurm_command(self, gpu_recommendation: Dict, 
                             script_name: str = "train.py") -> str:
        """Generate SLURM command based on GPU recommendation"""
        
        if not gpu_recommendation["fits_on_single_gpu"]:
            return "# Multi-GPU training command (example):\n" + \
                   "# srun --nodes=2 --ntasks-per-node=4 --gres=gpu:4 python train.py"
        
        gpu_name = gpu_recommendation["recommended_gpu"]
        gpus_needed = gpu_recommendation["gpus_needed"]
        
        # Find nodes with this GPU type
        gpu_specs = self.gpu_specs[gpu_name]
        
        if gpus_needed == 1:
            return f"srun --gres=gpu:{gpu_name}:1 python {script_name}"
        else:
            return f"srun --gres=gpu:{gpu_name}:{gpus_needed} python {script_name}"
    
    def print_memory_breakdown(self, model_name: str, memory_info: Dict):
        """Print detailed memory breakdown"""
        print(f"\n{'='*60}")
        print(f"📊 MEMORY BREAKDOWN: {model_name}")
        print(f"{'='*60}")
        
        print(f"{'Component':<20} {'Memory (MB)':<15} {'Percentage':<10}")
        print("-" * 45)
        
        components = [
            ("Parameters", memory_info["param_memory_mb"]),
            ("Gradients", memory_info["gradient_memory_mb"]),
            ("Optimizer", memory_info["optimizer_memory_mb"]),
            ("Batch Data", memory_info["batch_memory_mb"]),
            ("Activations", memory_info["activation_memory_mb"])
        ]
        
        total = memory_info["total_memory_mb"]
        
        for name, memory in components:
            percentage = (memory / total) * 100 if total > 0 else 0
            print(f"{name:<20} {memory:<15.1f} {percentage:<10.1f}%")
        
        print("-" * 45)
        print(f"{'TOTAL':<20} {total:<15.1f} {'100.0':<10}%")
        print(f"\nModel Parameters: {memory_info['total_params']:,}")
        print(f"Input Shape: {memory_info['input_shape']}")


def main():
    parser = argparse.ArgumentParser(description="Analyze GPU memory needs for PyTorch models")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--image-size", type=str, default="128", help="Image size (e.g., 128 or 128x128)")
    parser.add_argument("--out-neurons", type=int, default=200, help="Number of output neurons")
    parser.add_argument("--num-images", type=int, help="Number of images in dataset (required)")
    parser.add_argument("--num-neurons", type=int, help="Number of neurons in dataset (required)")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"], 
                       help="Data type for calculations")
    parser.add_argument("--models", type=str, nargs="+", default=["SimpleEncoder", "SimpleDecoder"],
                       help="Models to analyze")
    
    args = parser.parse_args()
    
    # Parse image size
    if "x" in args.image_size:
        image_size = tuple(map(int, args.image_size.split("x")))
    else:
        image_size = int(args.image_size)
    
    print("🔍 GPU Memory Analysis for Neurodecoders")
    print("=" * 60)
    
    analyzer = GPUMemoryAnalyzer(dtype=args.dtype)
    
    # Analyze models
    models_to_analyze = {}
    if "SimpleEncoder" in args.models:
        models_to_analyze["SimpleEncoder"] = SimpleEncoder(args.out_neurons)
    if "SimpleDecoder" in args.models:
        models_to_analyze["SimpleDecoder"] = SimpleDecoder(args.out_neurons, image_size if isinstance(image_size, int) else image_size[0])
    
    total_memory_mb = 0
    
    for model_name, model in models_to_analyze.items():
        memory_info = analyzer.estimate_model_memory(
            model, args.batch_size, image_size, args.out_neurons
        )
        analyzer.print_memory_breakdown(model_name, memory_info)
        total_memory_mb += memory_info["total_memory_mb"]
    
    # Dataset memory estimation
    print(f"\n{'='*60}")
    print("📁 DATASET ANALYSIS")
    print(f"{'='*60}")
    
    if args.num_images is not None and args.num_neurons is not None:
        dtype_size = getattr(torch, args.dtype).itemsize
        total_elements = args.num_images * args.num_neurons
        total_bytes = total_elements * dtype_size
        dataset_memory_mb = total_bytes / (1024 * 1024)
        print(f"User-specified dataset shape: ({args.num_images:,}, {args.num_neurons:,}) [images x neurons]")
        print(f"Per-sample memory: {args.num_neurons * dtype_size:.1f} bytes")
        print(f"Total dataset memory: {dataset_memory_mb:.1f} MB")
        total_memory_mb += dataset_memory_mb
        print("Note: Used user-specified shape for dataset memory estimate.")
    else:
        print("⚠️  Please specify both --num-images and --num-neurons to estimate dataset memory.")
    
    # GPU recommendations
    print(f"\n{'='*60}")
    print("🚀 GPU RECOMMENDATIONS")
    print(f"{'='*60}")
    
    gpu_recommendation = analyzer.recommend_gpu(total_memory_mb, args.batch_size)
    
    print(f"Total estimated memory: {total_memory_mb:.1f} MB ({total_memory_mb/1024:.1f} GB)")
    print(f"Fits on single GPU: {'✅ Yes' if gpu_recommendation['fits_on_single_gpu'] else '❌ No'}")
    
    if gpu_recommendation["fits_on_single_gpu"]:
        print(f"Recommended GPU: {gpu_recommendation['recommended_gpu']}")
        print(f"Memory utilization: {gpu_recommendation['memory_utilization_percent']:.1f}%")
        print(f"Parallelism strategy: {gpu_recommendation['parallelism_strategy']}")
    else:
        print(f"Recommended strategy: {gpu_recommendation['recommendation']}")
        print("Suggestions:")
        for suggestion in gpu_recommendation['suggestions']:
            print(f"  {suggestion}")
    
    # Generate SLURM command
    print(f"\n{'='*60}")
    print("⚡ SLURM COMMAND")
    print(f"{'='*60}")
    
    slurm_cmd = analyzer.generate_slurm_command(gpu_recommendation)
    print(slurm_cmd)
    
    # Additional suggestions
    if gpu_recommendation["fits_on_single_gpu"]:
        print(f"\n💡 OPTIMIZATION SUGGESTIONS:")
        for suggestion in gpu_recommendation['suggestions']:
            print(f"  {suggestion}")
    
    # Mixed precision analysis
    if args.dtype == "float32":
        mixed_precision_recommendation = analyzer.recommend_gpu(total_memory_mb, args.batch_size, use_mixed_precision=True)
        if mixed_precision_recommendation["fits_on_single_gpu"] and not gpu_recommendation["fits_on_single_gpu"]:
            print(f"\n🎯 MIXED PRECISION BENEFIT:")
            print(f"  With float16: {mixed_precision_recommendation['required_memory_mb']:.1f} MB")
            print(f"  Recommended GPU: {mixed_precision_recommendation['recommended_gpu']}")
            print(f"  Memory savings: {(1 - mixed_precision_recommendation['required_memory_mb'] / total_memory_mb) * 100:.1f}%")


if __name__ == "__main__":
    main() 