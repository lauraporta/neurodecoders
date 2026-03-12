"""
Training utilities for neural decoders.
"""

from typing import Any, Dict, Optional

import torch

from neurodecoders.core.base_lightning_module import BaseLightningModule
from neurodecoders.core.training_runner import TrainingConfig, create_trainer
from neurodecoders.data import NeuralDataModule
from neurodecoders.decoder.models import get_decoder_model


class DecoderLightningModule(BaseLightningModule):
    """
    PyTorch Lightning module for training neural decoders.

    Inherits from BaseLightningModule and adds decoder-specific functionality:
    - Special handling for diffusion models
    - Batch order (images, firing_rates) for decoder training
    - Model factory integration via get_decoder_model

    Note: Decoders predict images FROM firing rates (reverse of encoders).
    """

    def __init__(
        self,
        in_neurons: int,
        image_size: int = 32,
        learning_rate: float = 1e-4,
        loss_fn: str = "mse",
        optimizer_type: str = "adam",
        model_type: str = "simple",
        model_kwargs: Optional[Dict[str, Any]] = None,
        scheduler_type: str = "none",
        scheduler_step_size: int = 30,
        scheduler_gamma: float = 0.1,
        max_epochs: int = 100,
    ):
        """Initialize the decoder lightning module.

        Args:
            in_neurons: Number of input neurons (firing rate dimension)
            image_size: Output image size (assumed square)
            learning_rate: Learning rate for optimizer
            loss_fn: Loss function ('mse', 'l1', 'smooth_l1')
            optimizer_type: Optimizer type ('adam', 'adamw', 'sgd')
            model_type: Decoder model type ('simple', 'transformer', 'diffusion')
            model_kwargs: Additional model-specific parameters
            scheduler_type: LR scheduler type ('none', 'step', 'cosine', 'reduce_on_plateau')
            scheduler_step_size: Step size for step scheduler
            scheduler_gamma: Gamma for scheduler
            max_epochs: Maximum epochs (used for cosine scheduler)
        """
        # Build model using factory
        kwargs = model_kwargs or {}
        model = get_decoder_model(
            model_type=model_type,
            in_neurons=in_neurons,
            image_size=image_size,
            **kwargs,
        )

        # Convert scheduler_type for base class compatibility
        base_scheduler = (
            "plateau" if scheduler_type == "reduce_on_plateau" else scheduler_type
        )

        super().__init__(
            model=model,
            learning_rate=learning_rate,
            optimizer_type=optimizer_type,
            optimizer_config={"type": optimizer_type},
            loss_fn=loss_fn,
            scheduler_type=base_scheduler,
            scheduler_config={
                "step_size": scheduler_step_size,
                "gamma": scheduler_gamma,
            },
            max_epochs=max_epochs,
        )

        # Store decoder-specific attributes
        self.model_type = model_type
        self.is_diffusion = model_type == "diffusion"
        self.in_neurons = in_neurons
        self.image_size = image_size

    def forward(
        self, x: torch.Tensor, target_images: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through the decoder.

        Args:
            x: Input firing rates
            target_images: Target images (only used for diffusion models)

        Returns:
            Reconstructed images (or loss for diffusion models)
        """
        if self.is_diffusion:
            return self.model(x, target_images)
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Decoder training step: predict images from firing rates.

        Args:
            batch: Tuple of (images, firing_rates)
            batch_idx: Index of the batch

        Returns:
            Loss tensor
        """
        images, firing_rates = batch

        if self.is_diffusion:
            # Diffusion returns loss directly when target_images provided
            loss = self.model(firing_rates, images)
        else:
            pred = self(firing_rates)
            loss = self.loss_fn(pred, images)

        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Decoder validation step.

        Args:
            batch: Tuple of (images, firing_rates)
            batch_idx: Index of the batch

        Returns:
            Loss tensor
        """
        images, firing_rates = batch

        if self.is_diffusion:
            loss = self.model(firing_rates, images)
        else:
            pred = self(firing_rates)
            loss = self.loss_fn(pred, images)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Decoder test step.

        Args:
            batch: Tuple of (images, firing_rates)
            batch_idx: Index of the batch

        Returns:
            Loss tensor
        """
        images, firing_rates = batch

        if self.is_diffusion:
            loss = self.model(firing_rates, images)
        else:
            pred = self(firing_rates)
            loss = self.loss_fn(pred, images)

        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=False
        )
        return loss


def train_decoder(
    images,
    firing_rates,
    test_images=None,
    test_firing_rates=None,
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    optimizer: str = "adam",
    loss_fn: str = "mse",
    model_type: str = "simple",
    model_kwargs: Optional[dict] = None,
    scheduler: str = "none",
    scheduler_step_size: int = 30,
    scheduler_gamma: float = 0.1,
    num_workers: int = 0,
    pin_memory: bool = True,
    enable_mixed_precision: bool = True,
    enable_early_stopping: bool = True,
    early_stopping_patience: int = 50,
    enable_checkpointing: bool = True,
    mlflow_experiment_name: str = "neural_decoder",
    mlflow_run_name: Optional[str] = None,
):
    """
    Train a decoder using Lightning and shared data module.
    
    Args:
        images: Training images array
        firing_rates: Neural firing rates array
        test_images: Test images array (optional, recommended to prevent data leakage)
        test_firing_rates: Test firing rates array (optional, recommended to prevent data leakage)
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate
        optimizer: Optimizer type ('adam', 'adamw', 'sgd')
        loss_fn: Loss function ('mse', 'l1', 'smooth_l1')
        model_type: Type of decoder model ('simple', 'transformer', 'diffusion')
        model_kwargs: Additional model-specific parameters:
            For 'transformer': patch_size, embed_dim, num_heads, num_layers, mlp_ratio, dropout
            For 'diffusion': base_channels, channel_mults, timesteps, beta_start, beta_end
        scheduler: Learning rate scheduler ('none', 'step', 'cosine', 'reduce_on_plateau')
        scheduler_step_size: Step size for step scheduler
        scheduler_gamma: Gamma for step scheduler
        num_workers: Number of data loader workers
        pin_memory: Whether to pin memory for GPU transfer
        enable_mixed_precision: Enable 16-bit mixed precision
        enable_early_stopping: Enable early stopping
        early_stopping_patience: Patience for early stopping
        enable_checkpointing: Enable model checkpointing
        mlflow_experiment_name: MLflow experiment name
        mlflow_run_name: MLflow run name
    
    Returns:
        Tuple of (trainer, lightning_model, data_module)
    """
    data_module = NeuralDataModule(
        images=images,
        firing_rates=firing_rates,
        labels=None,
        test_images=test_images,
        test_firing_rates=test_firing_rates,
        test_labels=None,
        batch_size=batch_size,
        dataset_metadata={},
        use_memory_mapping=False,
        chunk_size=100,
        prefetch_factor=2,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    lightning_model = DecoderLightningModule(
        in_neurons=firing_rates.shape[1],
        image_size=int(images.shape[-1]),
        learning_rate=learning_rate,
        loss_fn=loss_fn,
        optimizer_type=optimizer,
        model_type=model_type,
        model_kwargs=model_kwargs,
        scheduler_type=scheduler,
        scheduler_step_size=scheduler_step_size,
        scheduler_gamma=scheduler_gamma,
        max_epochs=epochs,
    )

    # Use shared TrainingConfig and create_trainer infrastructure
    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer_type=optimizer,
        scheduler_type=scheduler,
        scheduler_step_size=scheduler_step_size,
        scheduler_gamma=scheduler_gamma,
        loss_fn=loss_fn,
        enable_mixed_precision=enable_mixed_precision,
        enable_early_stopping=enable_early_stopping,
        early_stopping_patience=early_stopping_patience,
        enable_checkpointing=enable_checkpointing,
        gradient_clip_val=1.0,  # Prevent NaN loss from gradient explosion
        enable_mlflow=True,  # MLflow setup handled by mlflow_training.py
        log_every_n_steps=10,
        num_workers=num_workers,
        pin_memory=pin_memory,
        model_name="decoder",
    )

    trainer = create_trainer(config)
    trainer.fit(lightning_model, data_module)

    return trainer, lightning_model, data_module
