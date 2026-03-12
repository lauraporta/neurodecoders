"""
Core infrastructure module for neurodecoders.

This module provides shared functionality for encoder and decoder training,
including base classes, callbacks, and training utilities that work with
both PyTorch Lightning and MLflow.
"""

from neurodecoders.core.base_lightning_module import BaseLightningModule
from neurodecoders.core.callbacks import MLflowMetricsCallback
from neurodecoders.core.training_runner import (
    TrainingConfig,
    UnfreezeCallback,
    create_standard_callbacks,
    create_trainer,
)

__all__ = [
    "BaseLightningModule",
    "MLflowMetricsCallback",
    "TrainingConfig",
    "UnfreezeCallback",
    "create_standard_callbacks",
    "create_trainer",
]
