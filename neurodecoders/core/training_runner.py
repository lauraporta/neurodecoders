"""
Training runner utilities for neurodecoders.

This module provides shared training infrastructure including:
- TrainingConfig dataclass for configuration
- create_trainer function for consistent trainer setup
- Common callback setup
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from neurodecoders.core.callbacks import MLflowMetricsCallback
from neurodecoders.paths import get_path


@dataclass
class TrainingConfig:
    """
    Configuration for training runs.

    This dataclass centralizes all training configuration parameters
    for both encoder and decoder training.
    """

    # Basic training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3

    # Optimizer settings
    optimizer_type: str = "adam"
    weight_decay: float = 0.0
    momentum: float = 0.9  # For SGD

    # Scheduler settings
    scheduler_type: str = "none"
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1
    scheduler_patience: int = 10  # For ReduceLROnPlateau

    # Loss function
    loss_fn: str = "mse"

    # Training features
    enable_mixed_precision: bool = True
    enable_early_stopping: bool = True
    early_stopping_patience: int = 50
    enable_checkpointing: bool = True
    gradient_clip_val: Optional[float] = None

    # Logging
    enable_mlflow: bool = True
    log_every_n_steps: int = 10
    enable_progress_bar: bool = True

    # Workers
    num_workers: int = 0
    pin_memory: bool = True

    # Checkpoint directory (relative to workspace)
    checkpoint_dir: str = "workspace/checkpoints"

    # Model name (used for checkpoint naming)
    model_name: str = "model"

    def to_optimizer_config(self) -> Dict[str, Any]:
        """Convert to optimizer config dict for BaseLightningModule."""
        return {
            "type": self.optimizer_type,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
        }

    def to_scheduler_config(self) -> Dict[str, Any]:
        """Convert to scheduler config dict for BaseLightningModule."""
        return {
            "type": self.scheduler_type,
            "step_size": self.scheduler_step_size,
            "gamma": self.scheduler_gamma,
            "patience": self.scheduler_patience,
        }


def create_standard_callbacks(
    config: TrainingConfig,
    checkpoint_filename_prefix: Optional[str] = None,
    additional_callbacks: Optional[List[Callback]] = None,
) -> List[Callback]:
    """
    Create standard callbacks based on training configuration.

    Args:
        config: Training configuration
        checkpoint_filename_prefix: Prefix for checkpoint files (defaults to config.model_name)
        additional_callbacks: Additional callbacks to include

    Returns:
        List of callbacks
    """
    callbacks: List[Callback] = []

    # MLflow metrics callback (handles learning rate logging)
    if config.enable_mlflow:
        callbacks.append(MLflowMetricsCallback(log_learning_rate=True))

    # Note: LearningRateMonitor requires a logger, so we only add it when MLflow is enabled
    # (MLflowMetricsCallback already logs LR). If you need LR monitoring without MLflow,
    # pass LearningRateMonitor via additional_callbacks along with a logger.

    # Early stopping
    if config.enable_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=config.early_stopping_patience,
                mode="min",
                verbose=True,
            )
        )

    # Model checkpointing
    if config.enable_checkpointing:
        prefix = checkpoint_filename_prefix or config.model_name
        callbacks.append(
            ModelCheckpoint(
                monitor="val_loss",
                dirpath=get_path(config.checkpoint_dir),
                filename=f"{prefix}-{{epoch:02d}}-{{val_loss:.4f}}",
                save_top_k=3,
                mode="min",
                verbose=True,
            )
        )

    # Add any additional callbacks
    if additional_callbacks:
        callbacks.extend(additional_callbacks)

    return callbacks


def create_trainer(
    config: TrainingConfig,
    callbacks: Optional[List[Callback]] = None,
    loggers: Optional[List[pl.loggers.Logger]] = None,
    **trainer_kwargs: Any,
) -> pl.Trainer:
    """
    Create a PyTorch Lightning trainer with standard configuration.

    Args:
        config: Training configuration
        callbacks: List of callbacks (if None, uses create_standard_callbacks)
        loggers: List of loggers (optional)
        **trainer_kwargs: Additional trainer arguments to override defaults

    Returns:
        Configured pl.Trainer instance
    """
    # Use provided callbacks or create standard ones
    if callbacks is None:
        callbacks = create_standard_callbacks(config)

    # Build trainer kwargs with defaults
    default_kwargs = {
        "max_epochs": config.epochs,
        "callbacks": callbacks,
        "logger": loggers or [],
        "enable_progress_bar": config.enable_progress_bar,
        "log_every_n_steps": config.log_every_n_steps,
        "accelerator": "auto",
        "devices": "auto",
        "strategy": "auto",
        "deterministic": False,
        "enable_checkpointing": config.enable_checkpointing,
        "precision": "16-mixed" if config.enable_mixed_precision else "32",
        "num_sanity_val_steps": 0,
    }

    # Add gradient clipping if specified
    if config.gradient_clip_val is not None:
        default_kwargs["gradient_clip_val"] = config.gradient_clip_val

    # Override with any custom kwargs
    default_kwargs.update(trainer_kwargs)

    return pl.Trainer(**default_kwargs)


class UnfreezeCallback(Callback):
    """Callback to unfreeze backbone layers at a specific epoch."""

    def __init__(self, unfreeze_epoch: int):
        """
        Initialize unfreeze callback.

        Args:
            unfreeze_epoch: Epoch at which to unfreeze backbone
        """
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Unfreeze backbone at specified epoch."""
        if trainer.current_epoch == self.unfreeze_epoch:
            print(f"Unfreezing backbone at epoch {self.unfreeze_epoch}")
            # Use the model's unfreeze_backbone method if available
            model = getattr(pl_module, "model", pl_module)
            if hasattr(model, "unfreeze_backbone"):
                model.unfreeze_backbone()
            elif hasattr(model, "backbone"):
                # Fallback for models with backbone attribute
                for param in model.backbone.parameters():
                    param.requires_grad = True


class TestEvaluationCallback(Callback):
    """
    Callback to evaluate test set during training epochs.

    This allows tracking test performance without data leakage,
    as the model is not trained on test data.
    """

    def __init__(self, test_dataloader):
        """
        Initialize test evaluation callback.

        Args:
            test_dataloader: DataLoader for test data
        """
        super().__init__()
        self.test_dataloader = test_dataloader
        self.test_losses: List[float] = []

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Evaluate test set at the end of each training epoch."""
        import numpy as np
        import torch

        if self.test_dataloader is None:
            return

        # Set model to evaluation mode
        pl_module.eval()

        test_losses = []
        with torch.no_grad():
            for batch in self.test_dataloader:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x, y = batch
                    # Move to device
                    x = x.to(pl_module.device)
                    y = y.to(pl_module.device)

                    # Forward pass
                    model = getattr(pl_module, "model", pl_module)
                    pred = model(x)
                    loss = pl_module.loss_fn(pred, y)
                    test_losses.append(loss.item())

        if test_losses:
            avg_test_loss = np.mean(test_losses)
            self.test_losses.append(avg_test_loss)

            # Store in the module
            if not hasattr(pl_module, "test_losses"):
                pl_module.test_losses = []
            pl_module.test_losses.append(avg_test_loss)

            # Log to PyTorch Lightning
            pl_module.log(
                "test_loss_epoch", avg_test_loss, on_step=False, on_epoch=True
            )

        # Set model back to training mode
        pl_module.train()
