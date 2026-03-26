"""
Base PyTorch Lightning module for neurodecoders.

This module provides a base class with shared functionality for both
encoder and decoder training, including:
- Loss function setup
- Optimizer configuration
- Learning rate scheduler configuration
- Training/validation step templates
"""

from abc import abstractmethod
from typing import Any, Dict, Literal, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn


# Type aliases for configuration
LossFnType = Literal["mse", "l1", "smooth_l1", "huber"]
OptimizerType = Literal["adam", "adamw", "sgd"]
SchedulerType = Literal["none", "step", "cosine", "plateau", "reduce_on_plateau"]


def create_loss_function(loss_fn: LossFnType) -> nn.Module:
    """
    Create a loss function from string specification.

    Args:
        loss_fn: Loss function type ("mse", "l1", "smooth_l1", "huber")

    Returns:
        PyTorch loss module

    Raises:
        ValueError: If loss_fn is not a supported type
    """
    loss_map = {
        "mse": nn.MSELoss,
        "l1": nn.L1Loss,
        "smooth_l1": nn.SmoothL1Loss,
        "huber": nn.HuberLoss,
    }
    if loss_fn not in loss_map:
        supported = ", ".join(loss_map.keys())
        raise ValueError(
            f"Unsupported loss function: {loss_fn}. Supported: {supported}"
        )
    return loss_map[loss_fn]()


class BaseLightningModule(pl.LightningModule):
    """
    Base PyTorch Lightning module with shared training infrastructure.

    This class provides common functionality for encoder and decoder modules:
    - Flexible optimizer configuration (Adam, AdamW, SGD)
    - Learning rate scheduler support (step, cosine, plateau)
    - Loss function setup
    - History tracking for train/val losses

    Subclasses must implement:
    - training_step(): Define forward pass and loss computation
    - forward(): Define model forward pass

    Optionally override:
    - validation_step(): Custom validation (default mirrors training_step)
    - test_step(): Custom test evaluation
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        optimizer_type: OptimizerType = "adam",
        optimizer_config: Optional[Dict[str, Any]] = None,
        loss_fn: LossFnType = "mse",
        scheduler_type: SchedulerType = "none",
        scheduler_config: Optional[Dict[str, Any]] = None,
        max_epochs: int = 100,
    ):
        """
        Initialize the base lightning module.

        Args:
            model: The PyTorch model to train
            learning_rate: Learning rate for optimizer
            optimizer_type: Type of optimizer ("adam", "adamw", "sgd")
            optimizer_config: Additional optimizer config (e.g., weight_decay)
            loss_fn: Loss function type ("mse", "l1", "smooth_l1", "huber")
            scheduler_type: LR scheduler type ("none", "step", "cosine", "plateau")
            scheduler_config: Additional scheduler config (e.g., step_size, gamma)
            max_epochs: Maximum training epochs (used for cosine scheduler)
        """
        super().__init__()
        # Don't save model in hyperparameters (too large)
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.optimizer_config = optimizer_config or {}
        self.loss_fn_name = loss_fn
        self.scheduler_type = scheduler_type
        self.scheduler_config = scheduler_config or {}
        self.max_epochs = max_epochs

        # Set up loss function
        self.loss_fn = create_loss_function(loss_fn)

        # Initialize history tracking
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.test_losses: list[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Model output
        """
        return self.model(x)

    @abstractmethod
    def training_step(
        self, batch: tuple, batch_idx: int
    ) -> torch.Tensor:
        """
        Training step - must be implemented by subclasses.

        Args:
            batch: Batch of training data
            batch_idx: Index of the batch

        Returns:
            Loss tensor
        """
        raise NotImplementedError("Subclasses must implement training_step")

    def validation_step(
        self, batch: tuple, batch_idx: int
    ) -> torch.Tensor:
        """
        Validation step - can be overridden by subclasses.

        Default implementation mirrors training_step pattern.

        Args:
            batch: Batch of validation data
            batch_idx: Index of the batch

        Returns:
            Loss tensor
        """
        # Default: same as training step pattern
        return self.training_step(batch, batch_idx)

    def on_train_epoch_end(self) -> None:
        """Store training loss at end of epoch."""
        loss = self.trainer.callback_metrics.get("train_loss")
        if loss is not None:
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            self.train_losses.append(float(loss))

    def on_validation_epoch_end(self) -> None:
        """Store validation loss at end of epoch."""
        loss = self.trainer.callback_metrics.get("val_loss")
        if loss is not None:
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            self.val_losses.append(float(loss))

    def configure_optimizers(self) -> Union[
        torch.optim.Optimizer,
        Dict[str, Any],
    ]:
        """
        Configure optimizer and learning rate scheduler.

        Returns:
            Optimizer or dict with optimizer and lr_scheduler
        """
        # Create optimizer
        optimizer = self._create_optimizer()

        # Return optimizer only if no scheduler
        if self.scheduler_type == "none":
            return optimizer

        # Create scheduler
        scheduler = self._create_scheduler(optimizer)

        # Handle ReduceLROnPlateau specially (needs monitor)
        if self.scheduler_type in ("plateau", "reduce_on_plateau"):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        weight_decay = self.optimizer_config.get("weight_decay", 0.0)

        if self.optimizer_type == "adam":
            return torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay,
            )
        elif self.optimizer_type == "adamw":
            return torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay,
            )
        elif self.optimizer_type == "sgd":
            momentum = self.optimizer_config.get("momentum", 0.9)
            return torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")

    def _create_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler based on configuration."""
        step_size = self.scheduler_config.get("step_size", 30)
        gamma = self.scheduler_config.get("gamma", 0.1)
        patience = self.scheduler_config.get("patience", 10)

        if self.scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=gamma,
            )
        elif self.scheduler_type == "cosine":
            # Use max_epochs from config, or try trainer
            t_max = self.max_epochs
            if hasattr(self, "trainer") and self.trainer is not None:
                t_max = self.trainer.max_epochs or t_max
            eta_min = self.learning_rate * self.scheduler_config.get(
                "eta_min_factor", 0.01
            )
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=eta_min,
            )
        elif self.scheduler_type in ("plateau", "reduce_on_plateau"):
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=gamma,
                patience=patience,
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_type}")

    def get_training_history(self) -> Dict[str, list[float]]:
        """
        Get training history for plotting.

        Returns:
            Dictionary with train_losses, val_losses, test_losses
        """
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "test_losses": self.test_losses,
        }
