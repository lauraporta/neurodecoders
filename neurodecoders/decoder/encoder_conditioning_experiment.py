#!/usr/bin/env python3
"""
Encoder Conditioning Experiment for Diffusion Decoders.

This script compares two approaches for conditioning diffusion decoders:

1. **Baseline**: Raw firing rates (5000d) → simple MLP → conditioning
2. **Encoder-Conditioned**: Firing rates → learned mapping → encoder embedding → conditioning

The hypothesis is that the encoder's learned representations capture "neurally relevant"
image features, providing a richer conditioning signal than raw firing rates.

Training flow for encoder-conditioned model:
1. Pre-trained encoder (frozen) extracts features from images: encoder(image) → 512d embedding
2. Neural-to-encoder mapper learns: firing_rates → predicted_encoder_embedding
3. Diffusion decoder is conditioned on predicted_encoder_embedding

At inference (only firing rates available):
- firing_rates → mapper → predicted_embedding → diffusion → reconstructed_image
"""

import argparse
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from neurodecoders.data.loading import (
    compute_and_apply_normalization,
    load_synthetic_split_data,
)
from neurodecoders.decoder.models import DiffusionDecoder, ConditionalUNet
from neurodecoders.input_optim.optimizer import load_encoder_from_mlflow
from neurodecoders.mlflow_utils.utils import setup_mlflow_experiment


# ==============================================================================
# Neural-to-Encoder Mapper
# ==============================================================================

class NeuralToEncoderMapper(nn.Module):
    """
    Maps firing rates to encoder embedding space.
    
    This network learns to predict what the encoder's representation would be
    for an image that produces the given firing rates.
    """
    
    def __init__(
        self,
        in_neurons: int,
        encoder_dim: int = 512,
        hidden_dims: Tuple[int, ...] = (1024, 512),
        dropout: float = 0.1,
    ):
        super().__init__()
        
        layers = []
        prev_dim = in_neurons
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, encoder_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, firing_rates: torch.Tensor) -> torch.Tensor:
        """Map firing rates to encoder embedding space."""
        return self.network(firing_rates)


# ==============================================================================
# Encoder-Conditioned Diffusion Decoder
# ==============================================================================

class EncoderConditionedDiffusionDecoder(nn.Module):
    """
    Diffusion decoder conditioned on encoder embeddings instead of raw firing rates.
    
    This uses a smaller conditioning dimension (encoder_dim=512) compared to
    raw firing rates (5000), but the embedding should be more informative.
    """
    
    def __init__(
        self,
        encoder_dim: int = 512,
        image_size: int = 32,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4),
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        output_scale: float = 2.5,
    ):
        super().__init__()
        self.image_size = image_size
        self.timesteps = timesteps
        self.output_scale = output_scale
        
        # Denoising U-Net - conditioned on encoder embedding
        self.unet = ConditionalUNet(
            in_channels=1,
            base_channels=base_channels,
            channel_mults=channel_mults,
            neural_dim=encoder_dim,  # Use encoder dimension instead of n_neurons
            time_dim=128,
        )
        
        # Setup noise schedule (same as standard DiffusionDecoder)
        self.register_buffer("betas", torch.linspace(beta_start, beta_end, timesteps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer(
            "alphas_cumprod_prev",
            F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        )
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - self.alphas_cumprod)
        )
        self.register_buffer(
            "posterior_variance",
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward diffusion: add noise to image."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
    
    def get_loss(
        self,
        x_start: torch.Tensor,
        encoder_cond: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the diffusion training loss.
        
        Args:
            x_start: Clean images (B, 1, H, W)
            encoder_cond: Encoder embeddings (B, encoder_dim)
            t: Optional timesteps (B,), randomly sampled if not provided
        
        Returns:
            MSE loss between predicted and actual noise
        """
        batch_size = x_start.size(0)
        device = x_start.device
        
        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size,), device=device)
        
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        
        predicted_noise = self.unet(x_noisy, t.float(), encoder_cond)
        
        return F.mse_loss(predicted_noise, noise)
    
    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: int,
        encoder_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Single denoising step."""
        batch_size = x.size(0)
        device = x.device
        
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.float32)
        
        predicted_noise = self.unet(x, t_tensor, encoder_cond)
        
        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        beta = self.betas[t]
        
        # Predict x_0
        x0_pred = (x - beta / torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha)
        x0_pred = torch.clamp(x0_pred, -self.output_scale, self.output_scale)
        
        if t > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(self.posterior_variance[t])
            x = x0_pred * torch.sqrt(self.alphas_cumprod_prev[t]) + \
                torch.sqrt(1 - self.alphas_cumprod_prev[t]) * noise
        else:
            x = x0_pred
        
        return x
    
    @torch.no_grad()
    def sample(
        self,
        encoder_cond: torch.Tensor,
        num_inference_steps: int = 50,
    ) -> torch.Tensor:
        """Generate images from encoder embeddings."""
        batch_size = encoder_cond.size(0)
        device = encoder_cond.device
        
        x = torch.randn(batch_size, 1, self.image_size, self.image_size, device=device)
        
        step_size = max(1, self.timesteps // num_inference_steps)
        timesteps = list(range(0, self.timesteps, step_size))[::-1]
        
        for t in tqdm(timesteps, desc="Sampling", leave=False):
            x = self.p_sample(x, t, encoder_cond)
        
        return torch.clamp(x, -self.output_scale, self.output_scale)


# ==============================================================================
# Training Functions
# ==============================================================================

def extract_encoder_features(
    encoder: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Extract encoder embeddings for all images."""
    encoder.eval()
    all_features = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting encoder features"):
            batch = images[i:i+batch_size].to(device)
            if batch.ndim == 3:
                batch = batch.unsqueeze(1)
            
            # Convert grayscale to RGB (encoder backbone expects 3 channels)
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)
            
            # Get features before the final firing_head
            features = encoder.backbone(batch)
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
            
            all_features.append(features.cpu())
    
    return torch.cat(all_features, dim=0)


def train_mapper(
    mapper: NeuralToEncoderMapper,
    firing_rates: torch.Tensor,
    encoder_features: torch.Tensor,
    val_firing_rates: torch.Tensor,
    val_encoder_features: torch.Tensor,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
) -> Dict[str, list]:
    """Train the neural-to-encoder mapper."""
    mapper = mapper.to(device)
    optimizer = torch.optim.AdamW(mapper.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    train_dataset = TensorDataset(firing_rates, encoder_features)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(val_firing_rates, val_encoder_features)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training
        mapper.train()
        train_losses = []
        for fr_batch, feat_batch in train_loader:
            fr_batch = fr_batch.to(device)
            feat_batch = feat_batch.to(device)
            
            optimizer.zero_grad()
            pred_feat = mapper(fr_batch)
            loss = F.mse_loss(pred_feat, feat_batch)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        mapper.eval()
        val_losses = []
        with torch.no_grad():
            for fr_batch, feat_batch in val_loader:
                fr_batch = fr_batch.to(device)
                feat_batch = feat_batch.to(device)
                pred_feat = mapper(fr_batch)
                loss = F.mse_loss(pred_feat, feat_batch)
                val_losses.append(loss.item())
        
        scheduler.step()
        
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        
        if (epoch + 1) % 10 == 0:
            print(f"  Mapper Epoch {epoch+1}/{epochs}: "
                  f"train_loss={history['train_loss'][-1]:.4f}, "
                  f"val_loss={history['val_loss'][-1]:.4f}")
    
    return history


def train_diffusion_model(
    model: nn.Module,
    images: torch.Tensor,
    conditioning: torch.Tensor,
    val_images: torch.Tensor,
    val_conditioning: torch.Tensor,
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    model_name: str = "model",
) -> Dict[str, list]:
    """Train a diffusion decoder."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    train_dataset = TensorDataset(images, conditioning)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(val_images, val_conditioning)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for img_batch, cond_batch in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}", leave=False):
            img_batch = img_batch.to(device)
            cond_batch = cond_batch.to(device)
            
            if img_batch.ndim == 3:
                img_batch = img_batch.unsqueeze(1)
            
            optimizer.zero_grad()
            loss = model.get_loss(img_batch, cond_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for img_batch, cond_batch in val_loader:
                img_batch = img_batch.to(device)
                cond_batch = cond_batch.to(device)
                if img_batch.ndim == 3:
                    img_batch = img_batch.unsqueeze(1)
                loss = model.get_loss(img_batch, cond_batch)
                val_losses.append(loss.item())
        
        scheduler.step()
        
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        
        if (epoch + 1) % 10 == 0:
            print(f"{model_name} Epoch {epoch+1}/{epochs}: "
                  f"train_loss={history['train_loss'][-1]:.4f}, "
                  f"val_loss={history['val_loss'][-1]:.4f}")
    
    return history


# ==============================================================================
# Evaluation Functions
# ==============================================================================

def evaluate_reconstruction(
    model: nn.Module,
    conditioning: torch.Tensor,
    target_images: torch.Tensor,
    encoder: nn.Module,
    target_firing_rates: torch.Tensor,
    device: torch.device,
    num_inference_steps: int = 50,
    batch_size: int = 16,
) -> Dict[str, float]:
    """Evaluate reconstruction quality."""
    model.eval()
    encoder.eval()
    
    all_generated = []
    all_pred_rates = []
    
    with torch.no_grad():
        for i in range(0, len(conditioning), batch_size):
            cond_batch = conditioning[i:i+batch_size].to(device)
            
            if hasattr(model, 'sample'):
                generated = model.sample(cond_batch, num_inference_steps=num_inference_steps)
            else:
                # Standard DiffusionDecoder uses different API
                batch_size_actual = cond_batch.size(0)
                x = torch.randn(batch_size_actual, 1, model.image_size, model.image_size, device=device)
                
                step_size = max(1, model.timesteps // num_inference_steps)
                timesteps = list(range(0, model.timesteps, step_size))[::-1]
                
                for t in timesteps:
                    t_tensor = torch.full((batch_size_actual,), t, device=device, dtype=torch.float32)
                    predicted_noise = model.unet(x, t_tensor, cond_batch)
                    
                    alpha = model.alphas[t]
                    alpha_cumprod = model.alphas_cumprod[t]
                    beta = model.betas[t]
                    
                    x0_pred = (x - beta / torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha)
                    x0_pred = torch.clamp(x0_pred, -model.output_scale, model.output_scale)
                    
                    if t > 0:
                        noise = torch.randn_like(x)
                        x = x0_pred * torch.sqrt(model.alphas_cumprod_prev[t]) + \
                            torch.sqrt(1 - model.alphas_cumprod_prev[t]) * noise
                    else:
                        x = x0_pred
                
                generated = torch.clamp(x, -model.output_scale, model.output_scale)
            
            all_generated.append(generated.cpu())
            
            # Get encoder predictions for neural correlation
            pred_rates = encoder(generated)
            all_pred_rates.append(pred_rates.cpu())
    
    generated = torch.cat(all_generated, dim=0)
    pred_rates = torch.cat(all_pred_rates, dim=0)
    
    # Ensure same shape
    if target_images.ndim == 3:
        target_images = target_images.unsqueeze(1)
    if generated.ndim == 3:
        generated = generated.unsqueeze(1)
    
    # Image metrics
    image_mse = F.mse_loss(generated, target_images).item()
    
    # Neural correlation
    pred_np = pred_rates.numpy()
    target_np = target_firing_rates.numpy()
    
    correlations = []
    for i in range(pred_np.shape[0]):
        corr = np.corrcoef(pred_np[i], target_np[i])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    
    return {
        'image_mse': image_mse,
        'neural_correlation_mean': np.mean(correlations) if correlations else 0.0,
        'neural_correlation_std': np.std(correlations) if correlations else 0.0,
        'neural_mse': F.mse_loss(pred_rates, target_firing_rates).item(),
    }


# ==============================================================================
# Main Experiment
# ==============================================================================

def run_experiment(config: Dict):
    """Run the encoder conditioning comparison experiment."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup MLflow
    setup_mlflow_experiment(
        config['experiment_name'],
        config.get('tracking_uri'),
    )
    
    with mlflow.start_run(run_name=config['run_name'], log_system_metrics=True):
        # Log configuration
        mlflow.log_params({
            'encoder_model_id': config['encoder_model_id'],
            'epochs': config['epochs'],
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
            'mapper_epochs': config['mapper_epochs'],
            'inference_steps': config['inference_steps'],
            'device': str(device),
        })
        
        # ================================================================
        # Load Data
        # ================================================================
        print("\n" + "="*60)
        print("Loading data...")
        print("="*60)
        
        # Load encoder
        print(f"Loading encoder from MLflow: {config['encoder_model_id']}")
        encoder_module = load_encoder_from_mlflow(config['encoder_model_id'])
        encoder_module = encoder_module.to(device)
        encoder_module.eval()
        encoder = encoder_module.model
        
        n_neurons = encoder.firing_head.out_features
        print(f"Encoder has {n_neurons} output neurons")
        
        # Load training data
        data_config = {
            'n_neurons': n_neurons,
            'dataset_type': 'cifar10',
            'sta_type': 'gabor,11,11',
        }
        
        train_images, train_firing, _, train_metadata = load_synthetic_split_data(
            data_config, split='train'
        )
        test_images, test_firing, _, test_metadata = load_synthetic_split_data(
            data_config, split='test'
        )
        
        print(f"Train: {len(train_images)} images, Test: {len(test_images)} images")
        
        # Apply normalization
        train_images, train_firing, test_images, test_firing, norm_stats = compute_and_apply_normalization(
            train_images, train_firing, test_images, test_firing
        )
        
        # Convert to tensors
        train_images = torch.tensor(train_images, dtype=torch.float32)
        train_firing = torch.tensor(train_firing, dtype=torch.float32)
        test_images = torch.tensor(test_images, dtype=torch.float32)
        test_firing = torch.tensor(test_firing, dtype=torch.float32)
        
        if train_images.ndim == 3:
            train_images = train_images.unsqueeze(1)
        if test_images.ndim == 3:
            test_images = test_images.unsqueeze(1)
        
        # Use validation split from training data
        n_val = min(500, len(train_images) // 10)
        val_images = train_images[-n_val:]
        val_firing = train_firing[-n_val:]
        train_images = train_images[:-n_val]
        train_firing = train_firing[:-n_val]
        
        print(f"Using {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
        
        # ================================================================
        # Extract Encoder Features
        # ================================================================
        print("\n" + "="*60)
        print("Extracting encoder features...")
        print("="*60)
        
        train_encoder_features = extract_encoder_features(encoder, train_images, device)
        val_encoder_features = extract_encoder_features(encoder, val_images, device)
        test_encoder_features = extract_encoder_features(encoder, test_images, device)
        
        encoder_dim = train_encoder_features.shape[1]
        print(f"Encoder feature dimension: {encoder_dim}")
        
        mlflow.log_param('encoder_dim', encoder_dim)
        
        # ================================================================
        # Train Neural-to-Encoder Mapper
        # ================================================================
        print("\n" + "="*60)
        print("Training neural-to-encoder mapper...")
        print("="*60)
        
        mapper = NeuralToEncoderMapper(
            in_neurons=n_neurons,
            encoder_dim=encoder_dim,
            hidden_dims=(1024, 512),
        )
        
        mapper_history = train_mapper(
            mapper,
            train_firing,
            train_encoder_features,
            val_firing,
            val_encoder_features,
            device,
            epochs=config['mapper_epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
        )
        
        mlflow.log_metric('mapper_final_train_loss', mapper_history['train_loss'][-1])
        mlflow.log_metric('mapper_final_val_loss', mapper_history['val_loss'][-1])
        
        # Get predicted encoder features for test set
        mapper.eval()
        with torch.no_grad():
            test_predicted_encoder_features = mapper(test_firing.to(device)).cpu()
        
        # ================================================================
        # Train Baseline Decoder (raw firing rates)
        # ================================================================
        print("\n" + "="*60)
        print("Training BASELINE decoder (raw firing rates)...")
        print("="*60)
        
        baseline_decoder = DiffusionDecoder(
            in_neurons=n_neurons,
            image_size=train_images.shape[-1],
            base_channels=config['base_channels'],
            channel_mults=config['channel_mults'],
            timesteps=config['timesteps'],
        )
        
        baseline_history = train_diffusion_model(
            baseline_decoder,
            train_images,
            train_firing,
            val_images,
            val_firing,
            device,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            model_name="Baseline",
        )
        
        mlflow.log_metric('baseline_final_train_loss', baseline_history['train_loss'][-1])
        mlflow.log_metric('baseline_final_val_loss', baseline_history['val_loss'][-1])
        
        # ================================================================
        # Train Encoder-Conditioned Decoder
        # ================================================================
        print("\n" + "="*60)
        print("Training ENCODER-CONDITIONED decoder...")
        print("="*60)
        
        encoder_decoder = EncoderConditionedDiffusionDecoder(
            encoder_dim=encoder_dim,
            image_size=train_images.shape[-1],
            base_channels=config['base_channels'],
            channel_mults=config['channel_mults'],
            timesteps=config['timesteps'],
        )
        
        # Train with TRUE encoder features (oracle conditioning)
        encoder_history = train_diffusion_model(
            encoder_decoder,
            train_images,
            train_encoder_features,  # Use true encoder features for training
            val_images,
            val_encoder_features,
            device,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            model_name="EncoderCond",
        )
        
        mlflow.log_metric('encoder_cond_final_train_loss', encoder_history['train_loss'][-1])
        mlflow.log_metric('encoder_cond_final_val_loss', encoder_history['val_loss'][-1])
        
        # ================================================================
        # Evaluate Both Models
        # ================================================================
        print("\n" + "="*60)
        print("Evaluating models...")
        print("="*60)
        
        # Limit test samples for faster evaluation
        n_test = min(config.get('n_test_samples', 200), len(test_images))
        test_subset_images = test_images[:n_test]
        test_subset_firing = test_firing[:n_test]
        test_subset_encoder_features = test_encoder_features[:n_test]
        test_subset_predicted_features = test_predicted_encoder_features[:n_test]
        
        # Baseline evaluation
        print("\nEvaluating baseline decoder...")
        baseline_metrics = evaluate_reconstruction(
            baseline_decoder,
            test_subset_firing,
            test_subset_images,
            encoder,
            test_subset_firing,
            device,
            num_inference_steps=config['inference_steps'],
        )
        
        for key, val in baseline_metrics.items():
            mlflow.log_metric(f'baseline_test_{key}', val)
        
        # Encoder-conditioned evaluation (with predicted features = realistic inference)
        print("\nEvaluating encoder-conditioned decoder (predicted features)...")
        encoder_cond_predicted_metrics = evaluate_reconstruction(
            encoder_decoder,
            test_subset_predicted_features,  # Use mapper predictions
            test_subset_images,
            encoder,
            test_subset_firing,
            device,
            num_inference_steps=config['inference_steps'],
        )
        
        for key, val in encoder_cond_predicted_metrics.items():
            mlflow.log_metric(f'encoder_cond_predicted_test_{key}', val)
        
        # Encoder-conditioned evaluation (with true features = oracle upper bound)
        print("\nEvaluating encoder-conditioned decoder (oracle features)...")
        encoder_cond_oracle_metrics = evaluate_reconstruction(
            encoder_decoder,
            test_subset_encoder_features,  # Use true encoder features
            test_subset_images,
            encoder,
            test_subset_firing,
            device,
            num_inference_steps=config['inference_steps'],
        )
        
        for key, val in encoder_cond_oracle_metrics.items():
            mlflow.log_metric(f'encoder_cond_oracle_test_{key}', val)
        
        # ================================================================
        # Generate Comparison Plots
        # ================================================================
        print("\n" + "="*60)
        print("Generating comparison plots...")
        print("="*60)
        
        # Training curves
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].plot(mapper_history['train_loss'], label='Train')
        axes[0].plot(mapper_history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Neural-to-Encoder Mapper')
        axes[0].legend()
        
        axes[1].plot(baseline_history['train_loss'], label='Train')
        axes[1].plot(baseline_history['val_loss'], label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Baseline Decoder')
        axes[1].legend()
        
        axes[2].plot(encoder_history['train_loss'], label='Train')
        axes[2].plot(encoder_history['val_loss'], label='Val')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Encoder-Conditioned Decoder')
        axes[2].legend()
        
        plt.tight_layout()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plt.savefig(f.name, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(f.name, "plots/training_curves.png")
        plt.close()
        
        # Metrics comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        models = ['Baseline', 'Enc-Cond\n(predicted)', 'Enc-Cond\n(oracle)']
        neural_corrs = [
            baseline_metrics['neural_correlation_mean'],
            encoder_cond_predicted_metrics['neural_correlation_mean'],
            encoder_cond_oracle_metrics['neural_correlation_mean'],
        ]
        neural_mses = [
            baseline_metrics['neural_mse'],
            encoder_cond_predicted_metrics['neural_mse'],
            encoder_cond_oracle_metrics['neural_mse'],
        ]
        
        colors = ['steelblue', 'coral', 'forestgreen']
        
        axes[0].bar(models, neural_corrs, color=colors)
        axes[0].set_ylabel('Neural Correlation')
        axes[0].set_title('Reconstruction Quality: Neural Correlation')
        axes[0].set_ylim(0, max(neural_corrs) * 1.2)
        
        axes[1].bar(models, neural_mses, color=colors)
        axes[1].set_ylabel('Neural MSE')
        axes[1].set_title('Reconstruction Quality: Neural MSE')
        
        plt.tight_layout()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plt.savefig(f.name, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(f.name, "plots/metrics_comparison.png")
        plt.close()
        
        # Visual reconstruction comparison
        n_display = min(5, n_test)
        
        # Generate samples for visualization
        baseline_decoder.eval()
        encoder_decoder.eval()
        
        with torch.no_grad():
            baseline_samples = []
            encoder_predicted_samples = []
            encoder_oracle_samples = []
            
            for i in range(n_display):
                # Baseline
                fr = test_subset_firing[i:i+1].to(device)
                x = torch.randn(1, 1, baseline_decoder.image_size, baseline_decoder.image_size, device=device)
                step_size = max(1, baseline_decoder.timesteps // config['inference_steps'])
                timesteps = list(range(0, baseline_decoder.timesteps, step_size))[::-1]
                for t in timesteps:
                    t_tensor = torch.full((1,), t, device=device, dtype=torch.float32)
                    predicted_noise = baseline_decoder.unet(x, t_tensor, fr)
                    alpha = baseline_decoder.alphas[t]
                    alpha_cumprod = baseline_decoder.alphas_cumprod[t]
                    beta = baseline_decoder.betas[t]
                    x0_pred = (x - beta / torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha)
                    x0_pred = torch.clamp(x0_pred, -baseline_decoder.output_scale, baseline_decoder.output_scale)
                    if t > 0:
                        noise = torch.randn_like(x)
                        x = x0_pred * torch.sqrt(baseline_decoder.alphas_cumprod_prev[t]) + \
                            torch.sqrt(1 - baseline_decoder.alphas_cumprod_prev[t]) * noise
                    else:
                        x = x0_pred
                baseline_samples.append(x.cpu())
                
                # Encoder-cond with predicted features
                pred_feat = test_subset_predicted_features[i:i+1].to(device)
                encoder_predicted_samples.append(encoder_decoder.sample(pred_feat, config['inference_steps']).cpu())
                
                # Encoder-cond with oracle features  
                true_feat = test_subset_encoder_features[i:i+1].to(device)
                encoder_oracle_samples.append(encoder_decoder.sample(true_feat, config['inference_steps']).cpu())
        
        fig, axes = plt.subplots(4, n_display, figsize=(3*n_display, 12))
        
        for j in range(n_display):
            # Original
            axes[0, j].imshow(test_subset_images[j, 0].numpy(), cmap='gray')
            axes[0, j].axis('off')
            if j == n_display // 2:
                axes[0, j].set_title('Original', fontsize=12)
            
            # Baseline
            axes[1, j].imshow(baseline_samples[j][0, 0].numpy(), cmap='gray')
            axes[1, j].axis('off')
            if j == n_display // 2:
                axes[1, j].set_title('Baseline', fontsize=12)
            
            # Encoder-cond predicted
            axes[2, j].imshow(encoder_predicted_samples[j][0, 0].numpy(), cmap='gray')
            axes[2, j].axis('off')
            if j == n_display // 2:
                axes[2, j].set_title('Enc-Cond (predicted)', fontsize=12)
            
            # Encoder-cond oracle
            axes[3, j].imshow(encoder_oracle_samples[j][0, 0].numpy(), cmap='gray')
            axes[3, j].axis('off')
            if j == n_display // 2:
                axes[3, j].set_title('Enc-Cond (oracle)', fontsize=12)
        
        plt.suptitle('Reconstruction Comparison', fontsize=14)
        plt.tight_layout()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plt.savefig(f.name, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(f.name, "plots/reconstructions.png")
        plt.close()
        
        # ================================================================
        # Summary
        # ================================================================
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        
        print(f"\nBaseline (raw firing rates):")
        print(f"  Neural correlation: {baseline_metrics['neural_correlation_mean']:.4f} "
              f"± {baseline_metrics['neural_correlation_std']:.4f}")
        print(f"  Neural MSE: {baseline_metrics['neural_mse']:.4f}")
        
        print(f"\nEncoder-Conditioned (predicted features):")
        print(f"  Neural correlation: {encoder_cond_predicted_metrics['neural_correlation_mean']:.4f} "
              f"± {encoder_cond_predicted_metrics['neural_correlation_std']:.4f}")
        print(f"  Neural MSE: {encoder_cond_predicted_metrics['neural_mse']:.4f}")
        
        print(f"\nEncoder-Conditioned (oracle features - upper bound):")
        print(f"  Neural correlation: {encoder_cond_oracle_metrics['neural_correlation_mean']:.4f} "
              f"± {encoder_cond_oracle_metrics['neural_correlation_std']:.4f}")
        print(f"  Neural MSE: {encoder_cond_oracle_metrics['neural_mse']:.4f}")
        
        improvement = (encoder_cond_predicted_metrics['neural_correlation_mean'] - 
                      baseline_metrics['neural_correlation_mean'])
        print(f"\nImprovement over baseline: {improvement:+.4f} correlation")
        
        mlflow.log_metric('improvement_over_baseline', improvement)
        
        print("\nExperiment completed. Check MLflow for detailed results.")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Encoder Conditioning Experiment for Diffusion Decoders'
    )
    
    parser.add_argument(
        '--encoder-model-id',
        type=str,
        required=True,
        help='MLflow run ID for the trained encoder'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='encoder_conditioning_experiment',
        help='MLflow experiment name'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='MLflow run name'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Training epochs for diffusion models'
    )
    parser.add_argument(
        '--mapper-epochs',
        type=int,
        default=50,
        help='Training epochs for neural-to-encoder mapper'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--base-channels',
        type=int,
        default=64,
        help='Base channels for U-Net'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=1000,
        help='Diffusion timesteps'
    )
    parser.add_argument(
        '--inference-steps',
        type=int,
        default=50,
        help='Inference steps for sampling'
    )
    parser.add_argument(
        '--n-test-samples',
        type=int,
        default=200,
        help='Number of test samples for evaluation'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Build config
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"enc_cond_experiment_{timestamp}"
    
    config = {
        'encoder_model_id': args.encoder_model_id,
        'experiment_name': args.experiment_name,
        'run_name': run_name,
        'epochs': args.epochs,
        'mapper_epochs': args.mapper_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'base_channels': args.base_channels,
        'channel_mults': (1, 2, 4),
        'timesteps': args.timesteps,
        'inference_steps': args.inference_steps,
        'n_test_samples': args.n_test_samples,
        'tracking_uri': os.environ.get('MLFLOW_TRACKING_URI'),
    }
    
    run_experiment(config)


if __name__ == '__main__':
    main()
