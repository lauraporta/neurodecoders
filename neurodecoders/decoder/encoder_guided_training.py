#!/usr/bin/env python3
"""
Encoder-Guided Decoder Training.

This module trains decoders using a frozen encoder in the loss computation.
The architecture is:
    firing_rates -> decoder (trainable) -> image -> encoder (frozen) -> predicted_firing_rates

The loss is computed on firing rates rather than pixels, encouraging the decoder
to generate images that drive neural activity matching the target.

Usage:
    python -m neurodecoders.decoder.encoder_guided_training \
        --encoder_run_id 9c5b51243d6248bc8967f8c5d224572d \
        --base_decoder_type simple \
        --loss_type mse \
        --epochs 100
"""

import argparse
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader, TensorDataset

from neurodecoders.config import get_base_path
from neurodecoders.data.loading import (
    compute_and_apply_normalization,
    load_synthetic_split_data,
)
from neurodecoders.decoder.models import (
    EncoderGuidedDecoder,
    create_encoder_guided_decoder,
)
from neurodecoders.input_optim.optimizer import load_encoder_from_mlflow
from neurodecoders.mlflow_utils.utils import (
    log_dataset_input_and_params,
    log_model_artifacts,
    log_training_config,
    log_training_metrics,
    setup_mlflow_experiment,
)
from neurodecoders.paths import get_path


# ==============================================================================
# Image Logging Callback
# ==============================================================================

class ImageLoggingCallback(Callback):
    """
    Callback to log sample generated images to MLflow during training.
    """
    
    def __init__(
        self,
        sample_firing_rates: torch.Tensor,
        sample_images: torch.Tensor,
        log_every_n_epochs: int = 10,
        n_samples: int = 5,
    ):
        """
        Args:
            sample_firing_rates: Sample firing rates to generate images from
            sample_images: Corresponding original images for comparison
            log_every_n_epochs: Log images every N epochs
            n_samples: Number of samples to log
        """
        super().__init__()
        self.sample_firing_rates = sample_firing_rates
        self.sample_images = sample_images
        self.log_every_n_epochs = log_every_n_epochs
        self.n_samples = min(n_samples, len(sample_firing_rates))
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Log sample images at the end of validation."""
        current_epoch = trainer.current_epoch
        
        # Only log at specified intervals and at the end
        if current_epoch % self.log_every_n_epochs != 0 and current_epoch != trainer.max_epochs - 1:
            return
        
        try:
            # Generate images
            device = pl_module.device
            firing_rates = self.sample_firing_rates[:self.n_samples].to(device)
            original_images = self.sample_images[:self.n_samples].cpu().numpy()
            
            pl_module.eval()
            with torch.no_grad():
                generated = pl_module.model.get_generated_images(firing_rates)
                generated_images = generated.cpu().numpy()
            pl_module.train()
            
            # Create comparison figure
            fig, axes = plt.subplots(2, self.n_samples, figsize=(3 * self.n_samples, 6))
            
            for i in range(self.n_samples):
                # Original image
                orig = original_images[i].squeeze()
                axes[0, i].imshow(orig, cmap='gray')
                axes[0, i].set_title(f'Original {i+1}')
                axes[0, i].axis('off')
                
                # Generated image
                gen = generated_images[i].squeeze()
                axes[1, i].imshow(gen, cmap='gray')
                axes[1, i].set_title(f'Generated {i+1}')
                axes[1, i].axis('off')
            
            axes[0, 0].set_ylabel('Original', fontsize=12)
            axes[1, 0].set_ylabel('Generated', fontsize=12)
            
            plt.suptitle(f'Epoch {current_epoch}', fontsize=14)
            plt.tight_layout()
            
            # Save and log to MLflow
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                plt.savefig(f.name, dpi=150, bbox_inches='tight')
                mlflow.log_artifact(f.name, f'images/epoch_{current_epoch:04d}.png')
                os.unlink(f.name)
            
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Could not log images at epoch {current_epoch}: {e}")


class FiringRateCorrelationCallback(Callback):
    """
    Callback to compute and log firing rate correlation during validation.
    """
    
    def __init__(
        self,
        sample_firing_rates: torch.Tensor,
        log_every_n_epochs: int = 5,
    ):
        super().__init__()
        self.sample_firing_rates = sample_firing_rates
        self.log_every_n_epochs = log_every_n_epochs
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Compute firing rate correlation."""
        current_epoch = trainer.current_epoch
        
        if current_epoch % self.log_every_n_epochs != 0:
            return
        
        try:
            device = pl_module.device
            firing_rates = self.sample_firing_rates.to(device)
            
            pl_module.eval()
            with torch.no_grad():
                predicted_rates = pl_module.model(firing_rates, return_images=False)
            pl_module.train()
            
            # Compute correlation per sample, then average
            firing_np = firing_rates.cpu().numpy()
            predicted_np = predicted_rates.cpu().numpy()
            
            correlations = []
            for i in range(len(firing_np)):
                corr = np.corrcoef(firing_np[i], predicted_np[i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            
            if correlations:
                mean_corr = np.mean(correlations)
                mlflow.log_metric('firing_rate_correlation', mean_corr, step=current_epoch)
                
        except Exception as e:
            print(f"Warning: Could not compute correlation at epoch {current_epoch}: {e}")


# ==============================================================================
# Lightning Module
# ==============================================================================

class EncoderGuidedDecoderLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for encoder-guided decoder training.
    """
    
    def __init__(
        self,
        model: EncoderGuidedDecoder,
        learning_rate: float = 1e-4,
        optimizer_type: str = 'adam',
        scheduler_type: str = 'cosine',
        max_epochs: int = 100,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        
        # Track losses
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
    
    def forward(self, firing_rates: torch.Tensor, return_images: bool = False):
        return self.model(firing_rates, return_images=return_images)
    
    def training_step(self, batch, batch_idx):
        images, firing_rates = batch
        
        # Compute loss using the model's loss function
        losses = self.model.compute_loss(firing_rates, target_images=images)
        
        # Log all loss components
        self.log('train_loss', losses['loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_fr_loss', losses['firing_rate_loss'], on_step=False, on_epoch=True)
        
        if 'pixel_loss' in losses:
            self.log('train_pixel_loss', losses['pixel_loss'], on_step=False, on_epoch=True)
        if 'correlation' in losses:
            self.log('train_correlation', losses['correlation'], on_step=False, on_epoch=True)
        
        return losses['loss']
    
    def validation_step(self, batch, batch_idx):
        images, firing_rates = batch
        
        losses = self.model.compute_loss(firing_rates, target_images=images)
        
        self.log('val_loss', losses['loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_fr_loss', losses['firing_rate_loss'], on_step=False, on_epoch=True)
        
        if 'pixel_loss' in losses:
            self.log('val_pixel_loss', losses['pixel_loss'], on_step=False, on_epoch=True)
        if 'correlation' in losses:
            self.log('val_correlation', losses['correlation'], on_step=False, on_epoch=True)
        
        return losses['loss']
    
    def test_step(self, batch, batch_idx):
        images, firing_rates = batch
        
        losses = self.model.compute_loss(firing_rates, target_images=images)
        
        self.log('test_loss', losses['loss'], on_step=False, on_epoch=True)
        self.log('test_fr_loss', losses['firing_rate_loss'], on_step=False, on_epoch=True)
        
        if 'pixel_loss' in losses:
            self.log('test_pixel_loss', losses['pixel_loss'], on_step=False, on_epoch=True)
        if 'correlation' in losses:
            self.log('test_correlation', losses['correlation'], on_step=False, on_epoch=True)
        
        return losses['loss']
    
    def on_train_epoch_end(self):
        loss = self.trainer.callback_metrics.get('train_loss')
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        if loss is not None:
            self.train_losses.append(loss)
    
    def on_validation_epoch_end(self):
        loss = self.trainer.callback_metrics.get('val_loss')
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        if loss is not None:
            self.val_losses.append(loss)
    
    def configure_optimizers(self):
        # Only optimize decoder parameters (encoder is frozen)
        params = self.model.decoder.parameters()
        
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
        
        if self.scheduler_type == 'none':
            return optimizer
        
        if self.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs, eta_min=self.learning_rate * 0.01
            )
        elif self.scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif self.scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'},
            }
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler_type}")
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


# ==============================================================================
# MLflow Metrics Callback
# ==============================================================================

class MLflowMetricsCallback(Callback):
    """Callback to log metrics to MLflow."""
    
    def __init__(self):
        self.logged_train_epochs = set()
        self.logged_val_epochs = set()
    
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.state.fn != 'fit':
            return
        
        epoch = trainer.current_epoch
        if epoch in self.logged_train_epochs:
            return
        self.logged_train_epochs.add(epoch)
        
        metrics = trainer.callback_metrics
        for key in ['train_loss', 'train_fr_loss', 'train_pixel_loss', 'train_correlation']:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                mlflow.log_metric(key, float(value), step=epoch)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.state.fn != 'fit':
            return
        
        epoch = trainer.current_epoch
        if epoch in self.logged_val_epochs:
            return
        self.logged_val_epochs.add(epoch)
        
        metrics = trainer.callback_metrics
        for key in ['val_loss', 'val_fr_loss', 'val_pixel_loss', 'val_correlation']:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                mlflow.log_metric(key, float(value), step=epoch)


# ==============================================================================
# Training Function
# ==============================================================================

def train_encoder_guided_decoder(
    train_images: np.ndarray,
    train_firing: np.ndarray,
    test_images: np.ndarray,
    test_firing: np.ndarray,
    encoder: nn.Module,
    config: Dict[str, Any],
) -> Tuple[pl.Trainer, EncoderGuidedDecoderLightningModule]:
    """
    Train an encoder-guided decoder.
    
    Args:
        train_images: Training images (N, C, H, W) - normalized
        train_firing: Training firing rates (N, N_neurons) - normalized
        test_images: Test images - normalized
        test_firing: Test firing rates - normalized
        encoder: Pre-trained encoder model
        config: Training configuration
    
    Returns:
        trainer, model
    """
    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(train_images, dtype=torch.float32),
        torch.tensor(train_firing, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(test_images, dtype=torch.float32),
        torch.tensor(test_firing, dtype=torch.float32),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True),
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True),
    )
    
    # Determine image size from data
    image_size = train_images.shape[-1]
    n_neurons = train_firing.shape[1]
    
    # Build decoder kwargs
    decoder_kwargs = {}
    if config['base_decoder_type'] == 'transformer':
        decoder_kwargs = {
            'patch_size': config.get('patch_size', 4),
            'embed_dim': config.get('embed_dim', 256),
            'num_heads': config.get('num_heads', 8),
            'num_layers': config.get('num_layers', 6),
            'mlp_ratio': config.get('mlp_ratio', 4.0),
            'dropout': config.get('transformer_dropout', 0.1),
        }
    
    # Create encoder-guided decoder
    model = create_encoder_guided_decoder(
        encoder=encoder,
        in_neurons=n_neurons,
        image_size=image_size,
        base_decoder_type=config['base_decoder_type'],
        loss_type=config['loss_type'],
        loss_weights=config.get('loss_weights'),
        tv_weight=config.get('tv_weight', 0.0),
        pixel_weight=config.get('pixel_weight', 0.0),
        **decoder_kwargs,
    )
    
    # Create Lightning module
    lightning_module = EncoderGuidedDecoderLightningModule(
        model=model,
        learning_rate=config['learning_rate'],
        optimizer_type=config.get('optimizer', 'adam'),
        scheduler_type=config.get('scheduler', 'cosine'),
        max_epochs=config['epochs'],
        weight_decay=config.get('weight_decay', 1e-5),
    )
    
    # Prepare sample data for callbacks
    n_vis_samples = min(8, len(test_images))
    vis_indices = np.random.choice(len(test_images), n_vis_samples, replace=False)
    sample_images = torch.tensor(test_images[vis_indices], dtype=torch.float32)
    sample_firing = torch.tensor(test_firing[vis_indices], dtype=torch.float32)
    
    # Setup callbacks
    callbacks = [
        MLflowMetricsCallback(),
        LearningRateMonitor(logging_interval='epoch'),
        ImageLoggingCallback(
            sample_firing_rates=sample_firing,
            sample_images=sample_images,
            log_every_n_epochs=config.get('log_images_every', 10),
            n_samples=min(5, n_vis_samples),
        ),
        FiringRateCorrelationCallback(
            sample_firing_rates=sample_firing,
            log_every_n_epochs=5,
        ),
    ]
    
    if config.get('enable_early_stopping', True):
        callbacks.append(
            EarlyStopping(
                monitor='val_loss',
                patience=config.get('early_stopping_patience', 30),
                mode='min',
            )
        )
    
    if config.get('enable_checkpointing', True):
        callbacks.append(
            ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=1,
                filename='best-{epoch:02d}-{val_loss:.4f}',
            )
        )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        accelerator='auto',
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=True,
        precision='16-mixed' if config.get('enable_mixed_precision', True) else 32,
        gradient_clip_val=config.get('gradient_clip_val', 1.0),
    )
    
    # Train
    trainer.fit(lightning_module, train_loader, val_loader)
    
    return trainer, lightning_module


# ==============================================================================
# Main Training Script
# ==============================================================================

def main(config: Dict[str, Any]):
    """Main training function."""
    print("=== Encoder-Guided Decoder Training ===")
    print(f"Base decoder type: {config['base_decoder_type']}")
    print(f"Loss type: {config['loss_type']}")
    print(f"Encoder run ID: {config['encoder_run_id']}")
    
    # Setup MLflow
    artifact_location = f"file://{get_base_path()}/mlruns"
    setup_mlflow_experiment(
        config['mlflow_experiment_name'],
        config.get('tracking_uri'),
        artifact_location=artifact_location,
    )
    
    with mlflow.start_run(run_name=config['mlflow_run_name'], log_system_metrics=True):
        # Log configuration
        log_training_config(config)
        mlflow.log_param('model_type', 'encoder_guided')
        mlflow.log_param('encoder_run_id', config['encoder_run_id'])
        
        # Load encoder from MLflow
        print(f"\nLoading encoder from run: {config['encoder_run_id']}")
        encoder = load_encoder_from_mlflow(config['encoder_run_id'])
        encoder.eval()
        print(f"Encoder loaded: {type(encoder).__name__}")
        
        # Load data
        print("\nLoading training data...")
        train_images, train_firing, _, train_meta = load_synthetic_split_data(config, split='train')
        print("\nLoading test data...")
        test_images, test_firing, _, test_meta = load_synthetic_split_data(config, split='test')
        
        print(f"Train images shape: {train_images.shape}")
        print(f"Train firing rates shape: {train_firing.shape}")
        print(f"Test images shape: {test_images.shape}")
        print(f"Test firing rates shape: {test_firing.shape}")
        
        # Normalize data (compute stats from training set, apply to both)
        print("\nApplying normalization...")
        train_images, train_firing, test_images, test_firing, norm_stats = compute_and_apply_normalization(
            train_images, train_firing, test_images, test_firing
        )
        
        print(f"After normalization:")
        print(f"  Train images range: [{train_images.min():.4f}, {train_images.max():.4f}]")
        print(f"  Train firing rates range: [{train_firing.min():.4f}, {train_firing.max():.4f}]")
        print(f"  Test images range: [{test_images.min():.4f}, {test_images.max():.4f}]")
        print(f"  Test firing rates range: [{test_firing.min():.4f}, {test_firing.max():.4f}]")
        
        # Log normalization stats
        mlflow.log_params({
            'norm_image_mean': norm_stats['image_mean'],
            'norm_image_std': norm_stats['image_std'],
            'norm_firing_mean': norm_stats.get('firing_mean', 0),
            'norm_firing_std': norm_stats.get('firing_std', 1),
        })
        
        # Train
        print("\nStarting training...")
        trainer, model = train_encoder_guided_decoder(
            train_images=train_images,
            train_firing=train_firing,
            test_images=test_images,
            test_firing=test_firing,
            encoder=encoder,
            config=config,
        )
        
        # Final evaluation
        print("\nRunning final evaluation on test set...")
        test_dataset = TensorDataset(
            torch.tensor(test_images, dtype=torch.float32),
            torch.tensor(test_firing, dtype=torch.float32),
        )
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        
        test_results = trainer.test(model, test_loader, verbose=False)
        if test_results:
            for key, value in test_results[0].items():
                mlflow.log_metric(f'final_{key}', value)
                print(f"  {key}: {value:.6f}")
        
        # Compute final firing rate correlation on full test set
        print("\nComputing final firing rate correlation...")
        device = next(model.parameters()).device
        model.eval()
        
        all_correlations = []
        with torch.no_grad():
            for batch_images, batch_firing in test_loader:
                batch_firing = batch_firing.to(device)
                predicted_rates = model.model(batch_firing, return_images=False)
                
                firing_np = batch_firing.cpu().numpy()
                pred_np = predicted_rates.cpu().numpy()
                
                for i in range(len(firing_np)):
                    corr = np.corrcoef(firing_np[i], pred_np[i])[0, 1]
                    if not np.isnan(corr):
                        all_correlations.append(corr)
        
        if all_correlations:
            final_corr = np.mean(all_correlations)
            final_corr_std = np.std(all_correlations)
            mlflow.log_metric('final_firing_rate_correlation', final_corr)
            mlflow.log_metric('final_firing_rate_correlation_std', final_corr_std)
            print(f"Final firing rate correlation: {final_corr:.4f} ± {final_corr_std:.4f}")
        
        # Save final comparison images
        print("\nSaving final comparison images...")
        try:
            n_final_samples = min(10, len(test_images))
            final_indices = np.random.choice(len(test_images), n_final_samples, replace=False)
            
            final_images = torch.tensor(test_images[final_indices], dtype=torch.float32).to(device)
            final_firing = torch.tensor(test_firing[final_indices], dtype=torch.float32).to(device)
            
            with torch.no_grad():
                generated_images = model.model.get_generated_images(final_firing)
            
            fig, axes = plt.subplots(3, n_final_samples, figsize=(2.5 * n_final_samples, 7.5))
            
            for i in range(n_final_samples):
                # Original
                orig = final_images[i].cpu().numpy().squeeze()
                axes[0, i].imshow(orig, cmap='gray')
                axes[0, i].set_title(f'Orig {i+1}', fontsize=8)
                axes[0, i].axis('off')
                
                # Generated
                gen = generated_images[i].cpu().numpy().squeeze()
                axes[1, i].imshow(gen, cmap='gray')
                axes[1, i].set_title(f'Gen {i+1}', fontsize=8)
                axes[1, i].axis('off')
                
                # Difference
                diff = np.abs(orig - gen)
                axes[2, i].imshow(diff, cmap='hot')
                axes[2, i].set_title(f'|Diff| {i+1}', fontsize=8)
                axes[2, i].axis('off')
            
            axes[0, 0].set_ylabel('Original', fontsize=10)
            axes[1, 0].set_ylabel('Generated', fontsize=10)
            axes[2, 0].set_ylabel('|Difference|', fontsize=10)
            
            plt.suptitle(f'Final Results - {config["base_decoder_type"]} decoder, {config["loss_type"]} loss', fontsize=12)
            plt.tight_layout()
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                plt.savefig(f.name, dpi=200, bbox_inches='tight')
                mlflow.log_artifact(f.name, 'final_comparison.png')
                os.unlink(f.name)
            plt.close(fig)
            print("Final comparison images saved.")
            
        except Exception as e:
            print(f"Warning: Could not save final images: {e}")
        
        # Save model
        print("\nSaving model...")
        model_dir = get_path('workspace/models/encoder_guided_decoders')
        os.makedirs(model_dir, exist_ok=True)
        
        model_name = f"encoder_guided_{config['base_decoder_type']}_{config['loss_type']}"
        model_path = os.path.join(model_dir, f'{model_name}.pth')
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        print(f"Model saved to {model_path}")
        
        # Log model with MLflow
        log_model_artifacts(
            model=model,
            model_name='encoder_guided_decoder',
            model_type='encoder_guided',
            dataset_info={
                'n_images': len(train_images),
                'n_neurons': train_firing.shape[1],
                'image_shape': train_images.shape[1:],
            },
            training_info={
                'epochs': config['epochs'],
                'learning_rate': config['learning_rate'],
                'base_decoder_type': config['base_decoder_type'],
                'loss_type': config['loss_type'],
                'encoder_run_id': config['encoder_run_id'],
            },
        )
    
    print("\n=== Training Complete ===")
    print("View results with: mlflow ui")


# ==============================================================================
# Argument Parser
# ==============================================================================

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train encoder-guided decoder',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        '--encoder_run_id',
        type=str,
        required=True,
        help='MLflow run ID of the pre-trained encoder',
    )
    
    # Decoder configuration
    parser.add_argument(
        '--base_decoder_type',
        type=str,
        default='simple',
        choices=['simple', 'transformer'],
        help='Base decoder architecture',
    )
    parser.add_argument(
        '--loss_type',
        type=str,
        default='mse',
        choices=['mse', 'poisson', 'correlation', 'combined'],
        help='Loss function type',
    )
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['none', 'cosine', 'step', 'reduce_on_plateau'])
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    
    # Data configuration
    parser.add_argument('--dataset_type', type=str, default='cifar10')
    parser.add_argument('--sta_type', type=str, default='gabor,11,11')
    parser.add_argument('--n_neurons', type=int, default=5000)
    parser.add_argument('--n_train_images', type=int, default=8000)
    parser.add_argument('--n_test_images', type=int, default=2000)
    
    # Transformer-specific parameters
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)
    parser.add_argument('--transformer_dropout', type=float, default=0.1)
    
    # MLflow configuration
    parser.add_argument('--mlflow_experiment_name', type=str, default='encoder_guided_decoder')
    parser.add_argument('--mlflow_run_name', type=str, default=None)
    parser.add_argument('--tracking_uri', type=str, default=None)
    
    # Other settings
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--enable_mixed_precision', type=bool, default=True)
    parser.add_argument('--enable_early_stopping', type=bool, default=True)
    parser.add_argument('--early_stopping_patience', type=int, default=30)
    parser.add_argument('--enable_checkpointing', type=bool, default=True)
    parser.add_argument('--log_images_every', type=int, default=10)
    
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    # Convert to config dict
    config = vars(args)
    
    # Set default run name if not provided
    if config['mlflow_run_name'] is None:
        config['mlflow_run_name'] = f"{config['base_decoder_type']}_{config['loss_type']}"
    
    main(config)
