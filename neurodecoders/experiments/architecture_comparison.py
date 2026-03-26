#!/usr/bin/env python3
"""
Architecture Comparison Experiment for Neural Image Reconstruction.

Compares three approaches:
1. Input optimization (gradient descent using frozen encoder)
2. Diffusion decoder (standalone)
3. Encoder-guided decoder (frozen encoder + trainable decoder)

Metrics collected:
- Training time
- Inference time for 2000 test images
- Memory usage
- Pixel correlation
- SSIM
- Neural consistency (re-stimulation with original Gabor filters)
"""

import argparse
import gc
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from neurodecoders.config import get_base_path, get_mlflow_tracking_uri
from neurodecoders.data.loading import (
    compute_and_apply_normalization,
    load_synthetic_split_data,
)
from neurodecoders.mlflow_utils.utils import (
    log_single_artifact,
    log_training_config,
    setup_mlflow_experiment,
)
from neurodecoders.paths import get_path


# =============================================================================
# Configuration
# =============================================================================

@dataclass  
class ExperimentConfig:
    """Configuration for the comparison experiment."""
    # Dataset config
    n_neurons: int = 5000
    n_train_images: int = 8000
    n_test_images: int = 2000
    dataset_type: str = "cifar10"
    sta_type: str = "gabor,11,11"
    
    # Encoder config (shared across approaches)
    encoder_type: str = "resnet_conv_only"
    encoder_lr: float = 0.001
    encoder_epochs: int = 100
    encoder_batch_size: int = 32
    
    # Diffusion decoder config
    diffusion_lr: float = 0.001
    diffusion_optimizer: str = "adamw"
    diffusion_scheduler: str = "cosine"
    diffusion_epochs: int = 100
    diffusion_batch_size: int = 32
    diffusion_embed_type: str = "simple"  # 'simple' or 'deep' neural embedding
    
    # Encoder-guided decoder config
    guided_decoder_type: str = "transformer"
    guided_loss_type: str = "correlation"
    guided_lr: float = 0.0001
    guided_epochs: int = 100
    guided_batch_size: int = 32
    guided_tv_weight: float = 0.0  # Total Variation loss weight for smoothness
    guided_pixel_weight: float = 0.0  # Auxiliary pixel-level MSE loss weight
    guided_optimizer: str = "adam"
    guided_scheduler: str = "cosine"
    guided_loss_weights_mse: float = 1.0  # MSE weight for combined loss
    guided_loss_weights_corr: float = 0.1  # Correlation weight for combined loss
    # Transformer-specific config
    guided_embed_dim: int = 256
    guided_num_layers: int = 6
    guided_num_heads: int = 8
    guided_patch_size: int = 4
    
    # Input optimization config
    input_optim_steps: int = 1000
    input_optim_lr: float = 0.1  # Best from sweep
    input_optim_loss: str = "poisson_mean"  # Best from sweep (vs mse)
    input_optim_blur_sigma: float = 1.5  # Best from sweep
    input_optim_scheduler: str = "none"  # Best from sweep (vs cosine)
    
    # MLflow config
    experiment_name: str = "architecture_comparison"
    
    # Run mode (for job array)
    mode: str = "all"  # "encoder", "input_optim", "diffusion", "guided", "all"
    encoder_run_id: Optional[str] = None  # Required for non-encoder modes


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


# =============================================================================
# Neural Consistency Metric (Re-stimulation)
# =============================================================================

def compute_neural_consistency(
    reconstructed_images: np.ndarray,
    original_firing_rates: np.ndarray,
    stas: np.ndarray,
    rf_coords: np.ndarray,
    device: torch.device,
    batch_size: int = 100,
) -> Dict[str, float]:
    """
    Compute neural consistency by re-stimulating with original Gabor filters.
    
    Args:
        reconstructed_images: (N, 1, H, W) reconstructed images
        original_firing_rates: (N, n_neurons) original firing rates
        stas: (n_neurons, rf_size, rf_size) Gabor filter patterns
        rf_coords: (n_neurons, 2) receptive field coordinates
        device: torch device
        batch_size: batch size for processing
        
    Returns:
        Dictionary with neural consistency metrics
    """
    from neurodecoders.synthetic.simulate_response import SimulateResponse
    
    n_images = reconstructed_images.shape[0]
    n_neurons = stas.shape[0]
    rf_size = stas.shape[1]
    image_height = reconstructed_images.shape[2]
    image_width = reconstructed_images.shape[3]
    
    # Convert to torch tensor
    images_tensor = torch.tensor(reconstructed_images, dtype=torch.float32)
    stas_tensor = torch.tensor(stas, dtype=torch.float32).to(device)
    
    # Compute firing rates for reconstructed images
    print("Computing firing rates for reconstructed images...")
    predicted_firing_rates = np.zeros((n_images, n_neurons))
    
    max_firing_rate = 100  # Same as in original simulation
    
    for batch_start in tqdm(range(0, n_images, batch_size), desc="Re-stimulation"):
        batch_end = min(batch_start + batch_size, n_images)
        batch_images = images_tensor[batch_start:batch_end].to(device)
        
        for n in range(n_neurons):
            x, y = rf_coords[n]
            # Extract patch at RF location
            patches = batch_images[:, 0, y:y+rf_size, x:x+rf_size]  # (batch, rf_size, rf_size)
            
            # Compute dot product with STA
            sta = stas_tensor[n]  # (rf_size, rf_size)
            dot = torch.sum(patches * sta, dim=(1, 2))  # (batch,)
            
            # Apply ELU and normalize
            response = torch.nn.functional.elu(dot)
            response = response / (response.max() + 1e-6) * max_firing_rate
            response = torch.clamp(response, min=0.0)
            
            predicted_firing_rates[batch_start:batch_end, n] = response.cpu().numpy()
    
    # Compute correlations
    correlations = []
    for i in range(n_images):
        corr, _ = pearsonr(original_firing_rates[i], predicted_firing_rates[i])
        if not np.isnan(corr):
            correlations.append(corr)
    
    correlations = np.array(correlations)
    
    return {
        "neural_consistency_mean": float(np.mean(correlations)),
        "neural_consistency_std": float(np.std(correlations)),
        "neural_consistency_median": float(np.median(correlations)),
        "neural_consistency_min": float(np.min(correlations)),
        "neural_consistency_max": float(np.max(correlations)),
    }


# =============================================================================
# Pixel Metrics
# =============================================================================

def compute_pixel_metrics(
    original_images: np.ndarray,
    reconstructed_images: np.ndarray,
) -> Dict[str, float]:
    """Compute pixel-level reconstruction metrics."""
    n_images = original_images.shape[0]
    
    correlations = []
    ssim_values = []
    mse_values = []
    
    for i in range(n_images):
        orig = original_images[i].squeeze()
        recon = reconstructed_images[i].squeeze()
        
        # Pixel correlation
        corr, _ = pearsonr(orig.flatten(), recon.flatten())
        if not np.isnan(corr):
            correlations.append(corr)
        
        # SSIM
        data_range = max(orig.max() - orig.min(), recon.max() - recon.min())
        if data_range > 0:
            ssim_val = ssim(orig, recon, data_range=data_range)
            ssim_values.append(ssim_val)
        
        # MSE
        mse = np.mean((orig - recon) ** 2)
        mse_values.append(mse)
    
    return {
        "pixel_correlation_mean": float(np.mean(correlations)),
        "pixel_correlation_std": float(np.std(correlations)),
        "pixel_correlation_median": float(np.median(correlations)),
        "ssim_mean": float(np.mean(ssim_values)),
        "ssim_std": float(np.std(ssim_values)),
        "ssim_median": float(np.median(ssim_values)),
        "mse_mean": float(np.mean(mse_values)),
        "mse_std": float(np.std(mse_values)),
    }


# =============================================================================
# Visualization
# =============================================================================

def create_comparison_grid(
    original_images: np.ndarray,
    reconstructions: Dict[str, np.ndarray],
    n_samples: int = 8,
    output_path: str = "comparison_grid.png",
) -> str:
    """Create a grid comparing original images with reconstructions from each method."""
    methods = list(reconstructions.keys())
    n_rows = n_samples
    n_cols = len(methods) + 1  # +1 for original
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))
    
    # Column headers
    col_titles = ["Original"] + methods
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight='bold')
    
    for row in range(n_samples):
        # Original
        axes[row, 0].imshow(original_images[row].squeeze(), cmap='gray')
        axes[row, 0].axis('off')
        
        # Reconstructions
        for col, method in enumerate(methods, start=1):
            axes[row, col].imshow(reconstructions[method][row].squeeze(), cmap='gray')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_metrics_comparison_plot(
    metrics: Dict[str, Dict[str, float]],
    output_path: str = "metrics_comparison.png",
) -> str:
    """Create bar plots comparing metrics across methods."""
    methods = list(metrics.keys())
    
    # Select key metrics to plot
    metric_names = [
        ("pixel_correlation_mean", "Pixel Correlation"),
        ("ssim_mean", "SSIM"),
        ("neural_consistency_mean", "Neural Consistency"),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for ax, (metric_key, metric_label) in zip(axes, metric_names):
        values = [metrics[m].get(metric_key, 0) for m in methods]
        errors = [metrics[m].get(metric_key.replace("_mean", "_std"), 0) for m in methods]
        
        bars = ax.bar(methods, values, yerr=errors, capsize=5, color=['#2196F3', '#4CAF50', '#FF9800'])
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_timing_comparison_plot(
    timing: Dict[str, Dict[str, float]],
    output_path: str = "timing_comparison.png",
) -> str:
    """Create bar plot comparing timing across methods."""
    methods = list(timing.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Training time
    train_times = [timing[m].get("training_time_seconds", 0) for m in methods]
    axes[0].bar(methods, train_times, color=['#2196F3', '#4CAF50', '#FF9800'])
    axes[0].set_ylabel("Time (seconds)")
    axes[0].set_title("Training Time")
    for i, v in enumerate(train_times):
        axes[0].text(i, v + max(train_times)*0.02, f'{v:.1f}s', ha='center')
    
    # Inference time (per 2000 images)
    inference_times = [timing[m].get("inference_time_seconds", 0) for m in methods]
    axes[1].bar(methods, inference_times, color=['#2196F3', '#4CAF50', '#FF9800'])
    axes[1].set_ylabel("Time (seconds)")
    axes[1].set_title("Inference Time (2000 images)")
    for i, v in enumerate(inference_times):
        axes[1].text(i, v + max(inference_times)*0.02, f'{v:.1f}s', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


# =============================================================================
# Phase 1: Train Encoder
# =============================================================================

def train_encoder(config: ExperimentConfig) -> Tuple[str, float]:
    """Train encoder and return run_id and training time."""
    from neurodecoders.data import NeuralDataModule
    from neurodecoders.encoder.models import ResNetConvOnly
    from neurodecoders.encoder.training import train_encoder as _train_encoder
    from neurodecoders.mlflow_utils.utils import log_model_artifacts
    
    print("\n" + "="*80)
    print("PHASE 1: Training Encoder")
    print("="*80)
    
    # Load data
    data_config = {
        "n_neurons": config.n_neurons,
        "n_train_images": config.n_train_images,
        "n_test_images": config.n_test_images,
        "dataset_type": config.dataset_type,
        "sta_type": config.sta_type,
    }
    
    train_images, train_firing, train_labels, train_meta = load_synthetic_split_data(
        data_config, split="train"
    )
    test_images, test_firing, test_labels, test_meta = load_synthetic_split_data(
        data_config, split="test"
    )
    
    print(f"Train data: {train_images.shape}, {train_firing.shape}")
    print(f"Test data: {test_images.shape}, {test_firing.shape}")
    
    # Create data module
    data_module = NeuralDataModule(
        images=train_images,
        firing_rates=train_firing,
        labels=train_labels,
        test_images=test_images,
        test_firing_rates=test_firing,
        test_labels=test_labels,
        batch_size=config.encoder_batch_size,
        dataset_metadata=train_meta,
    )
    
    # Create model
    model = ResNetConvOnly(out_neurons=config.n_neurons, freeze_backbone=False)
    
    # Setup MLflow
    artifact_location = f"file://{get_base_path()}/mlruns"
    setup_mlflow_experiment(config.experiment_name, artifact_location=artifact_location)
    
    run_name = f"encoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name, log_system_metrics=True) as run:
        run_id = run.info.run_id
        
        mlflow.log_params({
            "phase": "encoder_training",
            "model_type": config.encoder_type,
            "learning_rate": config.encoder_lr,
            "epochs": config.encoder_epochs,
            "batch_size": config.encoder_batch_size,
            "n_neurons": config.n_neurons,
            "n_train_images": config.n_train_images,
            "n_test_images": config.n_test_images,
            "dataset_type": config.dataset_type,
            "sta_type": config.sta_type,
        })
        
        # Train
        start_time = time.time()
        
        fold_results = _train_encoder(
            model=model,
            data_module=data_module,
            model_name="resnet_conv_only_encoder",
            learning_rate=config.encoder_lr,
            epochs=config.encoder_epochs,
            optimizer_config={"type": "adam"},
            loss_fn="mse",
            scheduler_config={"type": "plateau", "step_size": 30, "gamma": 0.1},
            enable_mlflow=False,  # We're handling MLflow ourselves
            n_folds=1,
            enable_mixed_precision=True,
            enable_early_stopping=True,
            early_stopping_patience=20,
            enable_checkpointing=True,
        )
        
        training_time = time.time() - start_time
        
        mlflow.log_metric("training_time_seconds", training_time)
        
        # Get the trained model
        if fold_results:
            trainer, trained_model, _ = fold_results[0]
            
            # Log model
            log_model_artifacts(
                model=trained_model.model if hasattr(trained_model, 'model') else trained_model,
                model_name="encoder_model",
                model_type="encoder",
                dataset_info={
                    "dataset_type": config.dataset_type,
                    "sta_type": config.sta_type,
                    "n_neurons": config.n_neurons,
                    "n_images": config.n_train_images,
                },
                training_info={
                    "epochs": config.encoder_epochs,
                    "learning_rate": config.encoder_lr,
                    "training_time": training_time,
                },
            )
        
        print(f"Encoder training completed in {training_time:.2f} seconds")
        print(f"Encoder run_id: {run_id}")
        
        return run_id, training_time


# =============================================================================
# Phase 2a: Input Optimization
# =============================================================================

def run_input_optimization(
    config: ExperimentConfig,
    encoder_run_id: str,
    encoder_training_time: float,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
    """Run input optimization and return reconstructions and metrics."""
    from neurodecoders.input_optim.optimizer import ImageOptimizer, OptimConfig, load_encoder_from_mlflow
    
    print("\n" + "="*80)
    print("PHASE 2a: Input Optimization")
    print("="*80)
    
    device = get_device()
    
    # Load encoder
    encoder = load_encoder_from_mlflow(encoder_run_id)
    encoder.to(device)
    encoder.eval()
    
    # Load test data
    data_config = {
        "n_neurons": config.n_neurons,
        "n_train_images": config.n_train_images,
        "n_test_images": config.n_test_images,
        "dataset_type": config.dataset_type,
        "sta_type": config.sta_type,
    }
    
    test_images, test_firing, _, test_meta = load_synthetic_split_data(data_config, split="test")
    
    # Normalize firing rates (same as training)
    train_images, train_firing, _, _ = load_synthetic_split_data(data_config, split="train")
    _, _, test_images_norm, test_firing_norm, _ = compute_and_apply_normalization(
        train_images, train_firing, test_images, test_firing
    )
    
    n_test = len(test_images_norm)
    image_size = test_images_norm.shape[-1]
    
    print(f"Running input optimization for {n_test} test images...")
    
    # Run optimization
    reconstructions = []
    start_time = time.time()
    
    for i in tqdm(range(n_test), desc="Input optimization"):
        target_rates = test_firing_norm[i].astype(np.float32)
        
        cfg = OptimConfig(
            image_size=image_size,
            channels=1,
            steps=config.input_optim_steps,
            lr=config.input_optim_lr,
            log_every=config.input_optim_steps + 1,  # Don't log intermediate
            seed=42 + i,
            loss=config.input_optim_loss,
            scheduler=config.input_optim_scheduler,
            blur_sigma=config.input_optim_blur_sigma,
        )
        
        optimizer = ImageOptimizer(encoder=encoder, target_rates=target_rates, config=cfg)
        img_np, _ = optimizer.optimize()
        reconstructions.append(img_np)
        
        # Clear memory periodically
        if i % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    inference_time = time.time() - start_time
    reconstructions = np.array(reconstructions)
    
    timing = {
        "training_time_seconds": encoder_training_time,
        "inference_time_seconds": inference_time,
        "total_time_seconds": encoder_training_time + inference_time,
    }
    
    memory = {
        "gpu_memory_mb": get_gpu_memory_mb(),
    }
    
    print(f"Input optimization completed in {inference_time:.2f} seconds")
    
    return reconstructions, timing, memory


# =============================================================================
# Phase 2b: Diffusion Decoder
# =============================================================================

def run_diffusion_decoder(
    config: ExperimentConfig,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
    """Train diffusion decoder and return reconstructions and metrics."""
    from neurodecoders.decoder.models import DiffusionDecoder
    from neurodecoders.decoder.training import train_decoder
    
    print("\n" + "="*80)
    print("PHASE 2b: Diffusion Decoder")
    print("="*80)
    
    device = get_device()
    
    # Load data
    data_config = {
        "n_neurons": config.n_neurons,
        "n_train_images": config.n_train_images,
        "n_test_images": config.n_test_images,
        "dataset_type": config.dataset_type,
        "sta_type": config.sta_type,
    }
    
    train_images, train_firing, _, _ = load_synthetic_split_data(data_config, split="train")
    test_images, test_firing, _, _ = load_synthetic_split_data(data_config, split="test")
    
    # Normalize
    train_images, train_firing, test_images_norm, test_firing_norm, _ = compute_and_apply_normalization(
        train_images, train_firing, test_images, test_firing
    )
    
    image_size = train_images.shape[-1]
    n_neurons = train_firing.shape[1]
    
    print(f"Training diffusion decoder...")
    print(f"  Image size: {image_size}")
    print(f"  Neurons: {n_neurons}")
    print(f"  Neural embed type: {config.diffusion_embed_type}")
    
    # Train
    start_time = time.time()
    
    trainer, lightning_model, _ = train_decoder(
        images=train_images,
        firing_rates=train_firing,
        test_images=test_images_norm,
        test_firing_rates=test_firing_norm,
        batch_size=config.diffusion_batch_size,
        epochs=config.diffusion_epochs,
        learning_rate=config.diffusion_lr,
        optimizer=config.diffusion_optimizer,
        model_type="diffusion",
        scheduler=config.diffusion_scheduler,
        enable_mixed_precision=True,
        enable_early_stopping=True,
        early_stopping_patience=30,
        enable_checkpointing=True,
        model_kwargs={"neural_embed_type": config.diffusion_embed_type},
    )
    
    training_time = time.time() - start_time
    
    # Inference
    print("Generating reconstructions for test set...")
    pytorch_model = lightning_model.model
    pytorch_model.to(device)
    pytorch_model.eval()
    
    start_time = time.time()
    
    test_firing_tensor = torch.tensor(test_firing_norm, dtype=torch.float32).to(device)
    
    reconstructions = []
    batch_size = 32
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_firing_tensor), batch_size), desc="Generating"):
            batch = test_firing_tensor[i:i+batch_size]
            # For diffusion, forward without target returns generated images
            generated = pytorch_model(batch, target_images=None)
            reconstructions.append(generated.cpu().numpy())
    
    inference_time = time.time() - start_time
    reconstructions = np.concatenate(reconstructions, axis=0)
    
    timing = {
        "training_time_seconds": training_time,
        "inference_time_seconds": inference_time,
        "total_time_seconds": training_time + inference_time,
    }
    
    memory = {
        "gpu_memory_mb": get_gpu_memory_mb(),
    }
    
    print(f"Diffusion training: {training_time:.2f}s, inference: {inference_time:.2f}s")
    
    return reconstructions, timing, memory


# =============================================================================
# Phase 2c: Encoder-Guided Decoder
# =============================================================================

def run_encoder_guided_decoder(
    config: ExperimentConfig,
    encoder_run_id: str,
    encoder_training_time: float,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
    """Train encoder-guided decoder and return reconstructions and metrics."""
    from neurodecoders.decoder.encoder_guided_training import train_encoder_guided_decoder
    from neurodecoders.decoder.models import create_encoder_guided_decoder
    from neurodecoders.input_optim.optimizer import load_encoder_from_mlflow
    
    print("\n" + "="*80)
    print("PHASE 2c: Encoder-Guided Decoder")
    print("="*80)
    
    device = get_device()
    
    # Load encoder
    encoder = load_encoder_from_mlflow(encoder_run_id)
    encoder.to(device)
    encoder.eval()
    
    # Load data
    data_config = {
        "n_neurons": config.n_neurons,
        "n_train_images": config.n_train_images,
        "n_test_images": config.n_test_images,
        "dataset_type": config.dataset_type,
        "sta_type": config.sta_type,
    }
    
    train_images, train_firing, _, _ = load_synthetic_split_data(data_config, split="train")
    test_images, test_firing, _, _ = load_synthetic_split_data(data_config, split="test")
    
    # Normalize
    train_images, train_firing, test_images_norm, test_firing_norm, _ = compute_and_apply_normalization(
        train_images, train_firing, test_images, test_firing
    )
    
    image_size = train_images.shape[-1]
    n_neurons = train_firing.shape[1]
    
    print(f"Training encoder-guided decoder...")
    print(f"  Base decoder: {config.guided_decoder_type}")
    print(f"  Loss type: {config.guided_loss_type}")
    print(f"  TV weight: {config.guided_tv_weight}")
    print(f"  Pixel weight: {config.guided_pixel_weight}")
    print(f"  Optimizer: {config.guided_optimizer}")
    print(f"  Scheduler: {config.guided_scheduler}")
    print(f"  LR: {config.guided_lr}")
    print(f"  Epochs: {config.guided_epochs}")
    if config.guided_loss_type == "combined":
        print(f"  Loss weights: mse={config.guided_loss_weights_mse}, corr={config.guided_loss_weights_corr}")
    if config.guided_decoder_type == "transformer":
        print(f"  Transformer: embed_dim={config.guided_embed_dim}, layers={config.guided_num_layers}, heads={config.guided_num_heads}, patch={config.guided_patch_size}")

    # Training config
    loss_weights = {"mse": config.guided_loss_weights_mse, "correlation": config.guided_loss_weights_corr}

    train_config = {
        "epochs": config.guided_epochs,
        "batch_size": config.guided_batch_size,
        "learning_rate": config.guided_lr,
        "base_decoder_type": config.guided_decoder_type,
        "loss_type": config.guided_loss_type,
        "loss_weights": loss_weights,
        "tv_weight": config.guided_tv_weight,
        "pixel_weight": config.guided_pixel_weight,
        "optimizer": config.guided_optimizer,
        "scheduler": config.guided_scheduler,
        "num_workers": 4,
        "pin_memory": True,
        "enable_early_stopping": True,
        "early_stopping_patience": 30,
        "enable_checkpointing": True,
        "enable_mixed_precision": True,
        # Transformer-specific
        "patch_size": config.guided_patch_size,
        "embed_dim": config.guided_embed_dim,
        "num_heads": config.guided_num_heads,
        "num_layers": config.guided_num_layers,
    }
    
    # Train
    start_time = time.time()
    
    trainer, lightning_model = train_encoder_guided_decoder(
        train_images=train_images,
        train_firing=train_firing,
        test_images=test_images_norm,
        test_firing=test_firing_norm,
        encoder=encoder,
        config=train_config,
    )
    
    decoder_training_time = time.time() - start_time
    
    # Inference
    print("Generating reconstructions for test set...")
    model = lightning_model.model
    model.to(device)
    model.eval()
    
    start_time = time.time()
    
    test_firing_tensor = torch.tensor(test_firing_norm, dtype=torch.float32).to(device)
    
    reconstructions = []
    batch_size = 32
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_firing_tensor), batch_size), desc="Generating"):
            batch = test_firing_tensor[i:i+batch_size]
            generated = model(batch, return_images=True)
            reconstructions.append(generated.cpu().numpy())
    
    inference_time = time.time() - start_time
    reconstructions = np.concatenate(reconstructions, axis=0)
    
    timing = {
        "training_time_seconds": encoder_training_time + decoder_training_time,
        "encoder_training_time_seconds": encoder_training_time,
        "decoder_training_time_seconds": decoder_training_time,
        "inference_time_seconds": inference_time,
        "total_time_seconds": encoder_training_time + decoder_training_time + inference_time,
    }
    
    memory = {
        "gpu_memory_mb": get_gpu_memory_mb(),
    }
    
    print(f"Guided decoder training: {decoder_training_time:.2f}s, inference: {inference_time:.2f}s")
    
    return reconstructions, timing, memory


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_comparison_experiment(config: ExperimentConfig):
    """Run the full comparison experiment."""
    
    print("\n" + "="*80)
    print("ARCHITECTURE COMPARISON EXPERIMENT")
    print("="*80)
    print(f"Mode: {config.mode}")
    print(f"Dataset: {config.dataset_type}, {config.sta_type}")
    print(f"Neurons: {config.n_neurons}, Train: {config.n_train_images}, Test: {config.n_test_images}")
    
    device = get_device()
    print(f"Device: {device}")
    
    # Setup output directory
    output_dir = get_path("workspace/plots/architecture_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup MLflow
    artifact_location = f"file://{get_base_path()}/mlruns"
    setup_mlflow_experiment(config.experiment_name, artifact_location=artifact_location)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Mode handling
    if config.mode == "encoder":
        # Just train encoder
        encoder_run_id, encoder_time = train_encoder(config)
        print(f"\nEncoder trained. Run ID: {encoder_run_id}")
        print(f"Use this for subsequent jobs: --encoder-run-id {encoder_run_id}")
        return
    
    elif config.mode in ["input_optim", "diffusion", "guided"]:
        # Single approach mode - requires encoder_run_id for input_optim and guided
        if config.mode in ["input_optim", "guided"] and not config.encoder_run_id:
            raise ValueError(f"--encoder-run-id required for mode '{config.mode}'")
        
        # Load test data for metrics
        data_config = {
            "n_neurons": config.n_neurons,
            "n_train_images": config.n_train_images,
            "n_test_images": config.n_test_images,
            "dataset_type": config.dataset_type,
            "sta_type": config.sta_type,
        }
        
        test_images, test_firing, _, test_meta = load_synthetic_split_data(data_config, split="test")
        train_images, train_firing, _, _ = load_synthetic_split_data(data_config, split="train")
        
        # Load Gabor filters for neural consistency
        test_data_path = test_meta.get("dataset_path", "")
        if test_data_path:
            test_data = np.load(test_data_path)
            stas = test_data["stas"]
            rf_coords = test_data["rf_coords"]
        else:
            # Try to find the test dataset file
            synthetic_dir = get_path("workspace/datasets/synthetic/test")
            files = [f for f in os.listdir(synthetic_dir) if f.endswith(".npz")]
            if files:
                test_data = np.load(os.path.join(synthetic_dir, files[0]))
                stas = test_data["stas"]
                rf_coords = test_data["rf_coords"]
            else:
                raise FileNotFoundError("Could not find test dataset with Gabor filters")
        
        # Normalize for comparison
        _, _, test_images_norm, test_firing_norm, _ = compute_and_apply_normalization(
            train_images, train_firing, test_images, test_firing
        )
        
        run_name = f"{config.mode}_{timestamp}"
        
        with mlflow.start_run(run_name=run_name, log_system_metrics=True):
            log_params = {
                "mode": config.mode,
                "n_neurons": config.n_neurons,
                "n_test_images": config.n_test_images,
                "dataset_type": config.dataset_type,
                "sta_type": config.sta_type,
            }

            # Log mode-specific hyperparameters
            if config.mode == "guided":
                log_params.update({
                    "guided_decoder_type": config.guided_decoder_type,
                    "guided_loss_type": config.guided_loss_type,
                    "guided_lr": config.guided_lr,
                    "guided_epochs": config.guided_epochs,
                    "guided_batch_size": config.guided_batch_size,
                    "guided_tv_weight": config.guided_tv_weight,
                    "guided_pixel_weight": config.guided_pixel_weight,
                    "guided_optimizer": config.guided_optimizer,
                    "guided_scheduler": config.guided_scheduler,
                    "guided_loss_weights_mse": config.guided_loss_weights_mse,
                    "guided_loss_weights_corr": config.guided_loss_weights_corr,
                })
                if config.guided_decoder_type == "transformer":
                    log_params.update({
                        "guided_embed_dim": config.guided_embed_dim,
                        "guided_num_layers": config.guided_num_layers,
                        "guided_num_heads": config.guided_num_heads,
                        "guided_patch_size": config.guided_patch_size,
                    })
            elif config.mode == "input_optim":
                log_params.update({
                    "input_optim_steps": config.input_optim_steps,
                    "input_optim_lr": config.input_optim_lr,
                    "input_optim_loss": config.input_optim_loss,
                    "input_optim_blur_sigma": config.input_optim_blur_sigma,
                    "input_optim_scheduler": config.input_optim_scheduler,
                })
            elif config.mode == "diffusion":
                log_params.update({
                    "diffusion_epochs": config.diffusion_epochs,
                    "diffusion_lr": config.diffusion_lr,
                    "diffusion_embed_type": config.diffusion_embed_type,
                })

            mlflow.log_params(log_params)
            
            if config.mode == "input_optim":
                # Get encoder training time from MLflow
                client = mlflow.tracking.MlflowClient()
                encoder_run = client.get_run(config.encoder_run_id)
                encoder_time = float(encoder_run.data.metrics.get("training_time_seconds", 0))
                
                reconstructions, timing, memory = run_input_optimization(
                    config, config.encoder_run_id, encoder_time
                )
                
            elif config.mode == "diffusion":
                reconstructions, timing, memory = run_diffusion_decoder(config)
                
            elif config.mode == "guided":
                client = mlflow.tracking.MlflowClient()
                encoder_run = client.get_run(config.encoder_run_id)
                encoder_time = float(encoder_run.data.metrics.get("training_time_seconds", 0))
                
                reconstructions, timing, memory = run_encoder_guided_decoder(
                    config, config.encoder_run_id, encoder_time
                )
            
            # Compute metrics
            print("\nComputing metrics...")
            pixel_metrics = compute_pixel_metrics(test_images_norm, reconstructions)
            neural_metrics = compute_neural_consistency(
                reconstructions, test_firing, stas, rf_coords, device
            )
            
            # Log all metrics
            all_metrics = {**pixel_metrics, **neural_metrics, **timing, **memory}
            mlflow.log_metrics(all_metrics)
            
            # Create plots
            comparison_path = os.path.join(output_dir, f"{config.mode}_samples_{timestamp}.png")
            create_comparison_grid(
                test_images_norm[:8],
                {config.mode: reconstructions[:8]},
                output_path=comparison_path,
            )
            mlflow.log_artifact(comparison_path)
            
            print(f"\nResults for {config.mode}:")
            for k, v in all_metrics.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    else:
        # Full comparison mode (all)
        raise ValueError(
            "mode='all' requires sequential job execution. "
            "Use sbatch job array with modes: encoder, input_optim, diffusion, guided"
        )


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Architecture Comparison Experiment")
    
    # Mode
    parser.add_argument("--mode", type=str, default="encoder",
                       choices=["encoder", "input_optim", "diffusion", "guided", "all"],
                       help="Which phase to run")
    parser.add_argument("--encoder-run-id", type=str, default=None,
                       help="MLflow run ID of trained encoder (required for input_optim and guided)")
    
    # Dataset
    parser.add_argument("--n-neurons", type=int, default=5000)
    parser.add_argument("--n-train-images", type=int, default=8000)
    parser.add_argument("--n-test-images", type=int, default=2000)
    parser.add_argument("--dataset-type", type=str, default="cifar10")
    parser.add_argument("--sta-type", type=str, default="gabor,11,11")
    
    # Encoder
    parser.add_argument("--encoder-epochs", type=int, default=100)
    parser.add_argument("--encoder-lr", type=float, default=0.001)
    
    # Diffusion
    parser.add_argument("--diffusion-epochs", type=int, default=100)
    parser.add_argument("--diffusion-lr", type=float, default=0.001)
    parser.add_argument("--diffusion-embed-type", type=str, default="simple",
                       choices=["simple", "deep"],
                       help="Neural embedding type: 'simple' (2-layer MLP) or 'deep' (4-layer with LayerNorm)")
    
    # Guided
    parser.add_argument("--guided-epochs", type=int, default=100)
    parser.add_argument("--guided-lr", type=float, default=0.0001)
    parser.add_argument("--guided-decoder-type", type=str, default="transformer",
                       choices=["simple", "transformer"],
                       help="Decoder architecture: 'simple' (CNN) or 'transformer'")
    parser.add_argument("--guided-tv-weight", type=float, default=0.0,
                       help="Total Variation loss weight for smoothness (0=disabled, try 0.001-0.1)")
    parser.add_argument("--guided-pixel-weight", type=float, default=0.0,
                       help="Auxiliary pixel-level MSE loss weight (0=disabled, try 0.1-0.5)")
    parser.add_argument("--guided-loss-type", type=str, default="correlation",
                       choices=["mse", "poisson", "correlation", "combined"],
                       help="Loss function for guided decoder")
    parser.add_argument("--guided-optimizer", type=str, default="adam",
                       choices=["adam", "adamw", "sgd"],
                       help="Optimizer for guided decoder")
    parser.add_argument("--guided-scheduler", type=str, default="cosine",
                       choices=["none", "cosine", "step", "reduce_on_plateau"],
                       help="LR scheduler for guided decoder")
    parser.add_argument("--guided-loss-weights-mse", type=float, default=1.0,
                       help="MSE weight for combined loss")
    parser.add_argument("--guided-loss-weights-corr", type=float, default=0.1,
                       help="Correlation weight for combined loss")
    parser.add_argument("--guided-batch-size", type=int, default=32)
    # Guided transformer-specific
    parser.add_argument("--guided-embed-dim", type=int, default=256,
                       help="Transformer embedding dimension")
    parser.add_argument("--guided-num-layers", type=int, default=6,
                       help="Number of transformer layers")
    parser.add_argument("--guided-num-heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--guided-patch-size", type=int, default=4,
                       help="Transformer patch size")
    
    # Input optim
    parser.add_argument("--input-optim-steps", type=int, default=1000)
    parser.add_argument("--input-optim-lr", type=float, default=0.1,
                       help="Learning rate (best from sweep: 0.1)")
    parser.add_argument("--input-optim-loss", type=str, default="poisson_mean",
                       help="Loss function: 'mse', 'poisson_mean', 'poisson_sum' (best: poisson_mean)")
    parser.add_argument("--input-optim-blur-sigma", type=float, default=1.5,
                       help="Gaussian blur sigma for gradient smoothing (best: 1.5)")
    parser.add_argument("--input-optim-scheduler", type=str, default="none",
                       help="LR scheduler: 'none', 'cosine', 'step', etc. (best: none)")
    
    # MLflow
    parser.add_argument("--experiment-name", type=str, default="architecture_comparison")
    
    args = parser.parse_args()
    
    config = ExperimentConfig(
        mode=args.mode,
        encoder_run_id=args.encoder_run_id,
        n_neurons=args.n_neurons,
        n_train_images=args.n_train_images,
        n_test_images=args.n_test_images,
        dataset_type=args.dataset_type,
        sta_type=args.sta_type,
        encoder_epochs=args.encoder_epochs,
        encoder_lr=args.encoder_lr,
        diffusion_epochs=args.diffusion_epochs,
        diffusion_lr=args.diffusion_lr,
        diffusion_embed_type=args.diffusion_embed_type,
        guided_epochs=args.guided_epochs,
        guided_lr=args.guided_lr,
        guided_decoder_type=args.guided_decoder_type,
        guided_tv_weight=args.guided_tv_weight,
        guided_pixel_weight=args.guided_pixel_weight,
        guided_loss_type=args.guided_loss_type,
        guided_optimizer=args.guided_optimizer,
        guided_scheduler=args.guided_scheduler,
        guided_loss_weights_mse=args.guided_loss_weights_mse,
        guided_loss_weights_corr=args.guided_loss_weights_corr,
        guided_batch_size=args.guided_batch_size,
        guided_embed_dim=args.guided_embed_dim,
        guided_num_layers=args.guided_num_layers,
        guided_num_heads=args.guided_num_heads,
        guided_patch_size=args.guided_patch_size,
        input_optim_steps=args.input_optim_steps,
        input_optim_lr=args.input_optim_lr,
        input_optim_loss=args.input_optim_loss,
        input_optim_blur_sigma=args.input_optim_blur_sigma,
        input_optim_scheduler=args.input_optim_scheduler,
        experiment_name=args.experiment_name,
    )
    
    run_comparison_experiment(config)


if __name__ == "__main__":
    main()
