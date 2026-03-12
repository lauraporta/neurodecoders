"""
Unified callbacks for PyTorch Lightning training with MLflow integration.

This module provides shared callbacks that can be used by both encoder
and decoder training pipelines.
"""

from typing import Optional, Set

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from neurodecoders.mlflow_utils.utils import log_training_metrics


class MLflowMetricsCallback(Callback):
    """
    Unified callback to log training and validation metrics to MLflow.

    This callback captures epoch-level metrics and logs them to MLflow with
    the epoch number as the step, enabling proper history tracking. It handles
    deduplication to ensure metrics are only logged once per epoch.

    Works with both encoder and decoder training pipelines.

    Attributes:
        log_learning_rate: Whether to log learning rate at each epoch
        metric_names: Custom metric names to log (defaults to train_loss, val_loss)
    """

    def __init__(
        self,
        log_learning_rate: bool = True,
        metric_names: Optional[list[str]] = None,
    ):
        """
        Initialize the MLflow metrics callback.

        Args:
            log_learning_rate: Whether to log learning rate changes
            metric_names: List of metric names to log. If None, logs
                         train_loss and val_loss by default.
        """
        super().__init__()
        self.log_learning_rate = log_learning_rate
        self.metric_names = metric_names or ["train_loss", "val_loss"]
        self.current_epoch: int = 0
        self.logged_train_epochs: Set[int] = set()
        self.logged_val_epochs: Set[int] = set()

    def _is_fitting(self, trainer: pl.Trainer) -> bool:
        """Check if trainer is in fit mode (not test/predict)."""
        if hasattr(trainer, "state") and hasattr(trainer.state, "fn"):
            return trainer.state.fn == "fit"
        return True

    def _log_metric_safely(
        self, metrics_dict: dict, metric_name: str, step: int
    ) -> Optional[float]:
        """
        Safely extract and log a metric to MLflow.

        Args:
            metrics_dict: Dictionary of metrics from trainer
            metric_name: Name of the metric to log
            step: Epoch number for logging

        Returns:
            The metric value if found and logged, None otherwise
        """
        value = metrics_dict.get(metric_name)
        if value is not None:
            if isinstance(value, torch.Tensor):
                value = value.item()
            try:
                log_training_metrics({metric_name: float(value)}, step=step)
                return float(value)
            except Exception as e:
                print(f"Warning: Could not log {metric_name} to MLflow: {e}")
        return None

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Log training metrics to MLflow at the end of each training epoch."""
        if not self._is_fitting(trainer):
            return

        self.current_epoch = trainer.current_epoch

        # Deduplicate: only log once per epoch
        if self.current_epoch in self.logged_train_epochs:
            return

        if trainer.callback_metrics:
            # Log train_loss
            train_loss = self._log_metric_safely(
                trainer.callback_metrics, "train_loss", self.current_epoch
            )

            # Store in module for history tracking
            if train_loss is not None:
                if not hasattr(pl_module, "train_losses"):
                    pl_module.train_losses = []
                pl_module.train_losses.append(train_loss)

        # Mark as logged
        self.logged_train_epochs.add(self.current_epoch)

        # Log learning rate if requested
        if self.log_learning_rate:
            self._log_learning_rate(pl_module)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Log validation metrics to MLflow at the end of each validation epoch."""
        if not self._is_fitting(trainer):
            return

        self.current_epoch = trainer.current_epoch

        # Deduplicate: only log once per epoch
        if self.current_epoch in self.logged_val_epochs:
            return

        if trainer.callback_metrics:
            # Log val_loss
            val_loss = self._log_metric_safely(
                trainer.callback_metrics, "val_loss", self.current_epoch
            )

            # Store in module for history tracking
            if val_loss is not None:
                if not hasattr(pl_module, "val_losses"):
                    pl_module.val_losses = []
                pl_module.val_losses.append(val_loss)

        # Mark as logged
        self.logged_val_epochs.add(self.current_epoch)

    def _log_learning_rate(self, pl_module: pl.LightningModule) -> None:
        """Log current learning rate if available."""
        try:
            if hasattr(pl_module, "optimizers") and pl_module.optimizers():
                optimizer = pl_module.optimizers()
                if hasattr(optimizer, "param_groups") and optimizer.param_groups:
                    current_lr = optimizer.param_groups[0]["lr"]
                    log_training_metrics(
                        {"learning_rate": float(current_lr)},
                        step=self.current_epoch,
                    )
        except Exception as e:
            # Learning rate logging is optional, don't fail on errors
            print(f"Warning: Could not log learning rate: {e}")


class EpochProgressCallback(Callback):
    """
    Simple callback to track epoch progress and store loss history.

    This is a lightweight alternative when MLflow logging is not needed,
    but loss history tracking is still desired.
    """

    def __init__(self):
        super().__init__()
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Store training loss at end of epoch."""
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            self.train_losses.append(float(loss))

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Store validation loss at end of epoch."""
        loss = trainer.callback_metrics.get("val_loss")
        if loss is not None:
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            self.val_losses.append(float(loss))
