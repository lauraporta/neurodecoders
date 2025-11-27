"""
Training utilities for neural decoders.
"""

from typing import List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from neurodecoders.data import NeuralDataModule
from neurodecoders.decoder.models import get_decoder_model

# MLflow utilities are no longer needed in this module
# They are handled by the calling script (mlflow_training.py)
from neurodecoders.paths import get_path


class MLflowMetricsCallback(Callback):
    """Custom callback to log metrics to MLflow during training."""

    def __init__(self):
        self.current_epoch = 0
        self.logged_train_epochs = set()
        self.logged_val_epochs = set()

    def on_train_epoch_end(self, trainer, pl_module):
        """Log training metrics at the end of each epoch."""
        try:
            import mlflow

            # Get the current epoch
            current_epoch = trainer.current_epoch

            # Only log during training phase, not during test
            if trainer.state.fn != "fit":
                return

            # Only log once per epoch
            if current_epoch in self.logged_train_epochs:
                return

            self.logged_train_epochs.add(current_epoch)

            # Get training loss from callback metrics
            train_loss = trainer.callback_metrics.get("train_loss")
            if train_loss is not None:
                if isinstance(train_loss, torch.Tensor):
                    train_loss = train_loss.item()
                mlflow.log_metric(
                    "train_loss", float(train_loss), step=current_epoch
                )

        except Exception as e:
            print(f"Warning: Could not log training metrics to MLflow: {e}")

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log validation metrics at the end of each epoch."""
        try:
            import mlflow

            # Get the current epoch
            current_epoch = trainer.current_epoch

            # Only log during training phase, not during test
            if trainer.state.fn != "fit":
                return

            # Only log once per epoch
            if current_epoch in self.logged_val_epochs:
                return

            self.logged_val_epochs.add(current_epoch)

            # Get validation loss from callback metrics
            val_loss = trainer.callback_metrics.get("val_loss")
            if val_loss is not None:
                if isinstance(val_loss, torch.Tensor):
                    val_loss = val_loss.item()
                mlflow.log_metric(
                    "val_loss", float(val_loss), step=current_epoch
                )

        except Exception as e:
            print(f"Warning: Could not log validation metrics to MLflow: {e}")


class DecoderLightningModule(pl.LightningModule):
    """Generic Lightning module for decoder training."""

    def __init__(
        self,
        in_neurons: int,
        image_size: int = 32,
        learning_rate: float = 1e-4,
        loss_fn: str = "mse",
        optimizer_type: str = "adam",
        model_type: str = "simple",
        model_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Build model kwargs
        kwargs = model_kwargs or {}
        
        # Use model factory for consistent interface
        self.model = get_decoder_model(
            model_type=model_type,
            in_neurons=in_neurons,
            image_size=image_size,
            **kwargs,
        )
        
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.model_type = model_type
        self.is_diffusion = model_type == "diffusion"

        if loss_fn == "mse":
            self.criterion = nn.MSELoss()
        elif loss_fn == "l1":
            self.criterion = nn.L1Loss()
        elif loss_fn == "smooth_l1":
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn}")

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def forward(self, x, target_images=None):
        if self.is_diffusion:
            # Diffusion model has different interface
            return self.model(x, target_images)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, firing_rates = batch
        
        if self.is_diffusion:
            # Diffusion returns loss directly when target_images provided
            loss = self.model(firing_rates, images)
        else:
            pred = self(firing_rates)
            loss = self.criterion(pred, images)
        
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, firing_rates = batch
        
        if self.is_diffusion:
            # Diffusion returns loss directly when target_images provided
            loss = self.model(firing_rates, images)
        else:
            pred = self(firing_rates)
            loss = self.criterion(pred, images)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, firing_rates = batch
        
        if self.is_diffusion:
            # Diffusion returns loss directly when target_images provided
            loss = self.model(firing_rates, images)
        else:
            pred = self(firing_rates)
            loss = self.criterion(pred, images)
        
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=False
        )
        return loss

    def on_train_epoch_end(self):
        loss = self.trainer.callback_metrics.get("train_loss")
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        if loss is not None:
            self.train_losses.append(loss)

    def on_validation_epoch_end(self):
        loss = self.trainer.callback_metrics.get("val_loss")
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        if loss is not None:
            self.val_losses.append(loss)

    def configure_optimizers(self):
        if self.optimizer_type == "adam":
            opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "adamw":
            opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "sgd":
            opt = torch.optim.SGD(
                self.parameters(), lr=self.learning_rate, momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")
        return opt


def train_decoder(
    images,
    firing_rates,
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    optimizer: str = "adam",
    loss_fn: str = "mse",
    model_type: str = "simple",
    model_kwargs: Optional[dict] = None,
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
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate
        optimizer: Optimizer type ('adam', 'adamw', 'sgd')
        loss_fn: Loss function ('mse', 'l1', 'smooth_l1')
        model_type: Type of decoder model ('simple', 'transformer', 'diffusion')
        model_kwargs: Additional model-specific parameters:
            For 'transformer': patch_size, embed_dim, num_heads, num_layers, mlp_ratio, dropout
            For 'diffusion': base_channels, channel_mults, timesteps, beta_start, beta_end
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
    )

    callbacks: List[pl.Callback] = [
        LearningRateMonitor(logging_interval="epoch"),
        MLflowMetricsCallback(),
    ]
    if enable_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                mode="min",
            )
        )
    if enable_checkpointing:
        callbacks.append(
            ModelCheckpoint(
                monitor="val_loss",
                dirpath=get_path("workspace/checkpoints"),
                filename="decoder-{epoch:02d}-{val_loss:.4f}",
                save_top_k=3,
                mode="min",
            )
        )

    # MLflow setup is handled by the calling script (mlflow_training.py)
    # No need to set up experiments or start runs here

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if enable_mixed_precision else "32",
        log_every_n_steps=10,
        enable_checkpointing=enable_checkpointing,
    )

    trainer.fit(lightning_model, data_module)

    return trainer, lightning_model, data_module
