"""
Generic training pipeline for neural encoder models.

This module provides a unified training interface that works with any
model architecture from the models module.
"""

from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger

from .mlflow_utils import log_encoder_experiment
from .models import (
    ResNetEncoder,
    SimpleEncoder,
    SimpleEncoderWithSkipConnection,
)
from .verification_callback import EncoderVerificationCallback


class EncoderLightningModule(pl.LightningModule):
    """
    Generic PyTorch Lightning module for training neural encoders.

    This module can work with any model architecture by accepting the model
    as a parameter instead of hardcoding it.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        optimizer_config: Optional[dict] = None,
        loss_fn: str = "mse",  # "mse", "l1", "smooth_l1", "huber"
        scheduler_config: Optional[dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.learning_rate = learning_rate
        self.optimizer_config = optimizer_config or {"type": "adam"}
        self.scheduler_config = scheduler_config or {"type": "none"}

        # Set up loss function
        if loss_fn == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_fn == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_fn == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss()
        elif loss_fn == "huber":
            self.loss_fn = nn.HuberLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn}")

        # Store training history for plotting
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)

        # Log training loss - epoch level only
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)

        # Log validation loss (epoch-level only)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)

        # Log test loss
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        # Calculate correlation coefficient between true and predicted firing
        # rates
        # Convert to numpy for correlation calculation
        y_np = y.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()

        # Calculate correlation for each neuron and take the mean
        correlations = []
        for i in range(y_np.shape[1]):
            corr = np.corrcoef(y_np[:, i], pred_np[:, i])[0, 1]
            if not np.isnan(corr):  # Handle NaN values
                correlations.append(corr)

        if correlations:
            mean_correlation = np.mean(correlations)
            self.log(
                "test_correlation",
                mean_correlation,
                on_step=False,
                on_epoch=True,
            )

        return loss

    def configure_optimizers(self):
        """
        Configure optimizers based on the optimizer_config.
        Supports different strategies for different model types.
        """
        if self.optimizer_config["type"] == "resnet_differential":
            # Different learning rates for backbone vs head (for ResNet models)
            backbone_params = []
            head_params = []

            # Separate parameters by component
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if "backbone" in name:
                        backbone_params.append(param)
                    else:
                        head_params.append(param)

            # Create parameter groups with different learning rates
            param_groups = [
                {
                    "params": head_params,
                    "lr": self.learning_rate,
                },
                {
                    "params": backbone_params,
                    "lr": self.learning_rate * 0.1,  # Lower LR for backbone
                },
            ]

            optimizer = torch.optim.Adam(param_groups)

        else:
            # Standard optimizer for all parameters
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
            )

        # Configure scheduler
        if self.scheduler_config["type"] == "reduce_lr_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                verbose=True,
                min_lr=1e-6,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }
        else:
            # No scheduler
            return optimizer

    def on_train_epoch_end(self):
        # Store losses for plotting
        train_loss = self.trainer.callback_metrics.get("train_loss_epoch", 0)
        val_loss = self.trainer.callback_metrics.get("val_loss", 0)

        if isinstance(train_loss, torch.Tensor):
            train_loss = train_loss.item()
        if isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.item()

        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)


class UnfreezeCallback(pl.Callback):
    """
    Callback to progressively unfreeze backbone layers during training.
    Useful for transfer learning with pre-trained models.
    """

    def __init__(self, unfreeze_epoch: int, num_layers: int = 2):
        self.unfreeze_epoch = unfreeze_epoch
        self.num_layers = num_layers

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.unfreeze_epoch:
            if hasattr(pl_module.model, "unfreeze_backbone"):
                print(f"\nUnfreezing backbone at epoch {self.unfreeze_epoch}")
                pl_module.model.unfreeze_backbone(num_layers=self.num_layers)


def train_encoder(
    model: nn.Module,
    data_module,
    model_name: str = "encoder",
    learning_rate: float = 1e-3,
    epochs: int = 30,
    optimizer_config: Optional[dict] = None,
    loss_fn: str = "mse",
    scheduler_config: Optional[dict] = None,
    callbacks: Optional[list] = None,
    enable_progress_bar: bool = True,
    log_every_n_steps: int = 50,
    unfreeze_epoch: Optional[int] = None,
    enable_mlflow: bool = True,
    mlflow_experiment_name: str = "neural_encoder",
    mlflow_run_name: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
    n_folds: int = 1,  # Default to 1 (no CV)
):
    """
    Generic training function that works with any model architecture.
    Supports both single training and k-fold cross-validation.

    Args:
        model: The neural network model to train
        data_module: Lightning data module with train/val/test dataloaders
        model_name: Name for the model (used in logging)
        learning_rate: Learning rate for training
        epochs: Number of training epochs
        optimizer_config: Dictionary specifying optimizer configuration
        loss_fn: Loss function type ("mse", "l1", "smooth_l1", "huber")
        scheduler_config: Dictionary specifying scheduler configuration
        callbacks: List of additional callbacks
        enable_progress_bar: Whether to show progress bar
        log_every_n_steps: Logging frequency
        unfreeze_epoch: Epoch to start unfreezing backbone (for transfer
        learning)
        enable_mlflow: Whether to enable MLflow logging
        mlflow_experiment_name: MLflow experiment name
        mlflow_run_name: MLflow run name
        mlflow_tracking_uri: MLflow tracking URI
        n_folds: Number of cross-validation folds (1 = no CV)

    Returns:
        If n_folds=1: (trainer, lightning_model, data_module)
        If n_folds>1: list of (trainer, lightning_model, data_module) tuples
    """

    # If n_folds=1, do regular training
    if n_folds == 1:
        return _train_single_fold(
            model=model,
            data_module=data_module,
            model_name=model_name,
            learning_rate=learning_rate,
            epochs=epochs,
            optimizer_config=optimizer_config,
            loss_fn=loss_fn,
            scheduler_config=scheduler_config,
            callbacks=callbacks,
            enable_progress_bar=enable_progress_bar,
            log_every_n_steps=log_every_n_steps,
            unfreeze_epoch=unfreeze_epoch,
            enable_mlflow=enable_mlflow,
            mlflow_experiment_name=mlflow_experiment_name,
            mlflow_run_name=mlflow_run_name,
            mlflow_tracking_uri=mlflow_tracking_uri,
        )

    # Otherwise, do k-fold cross-validation
    return _train_with_cv(
        model=model,
        data_module=data_module,
        model_name=model_name,
        n_folds=n_folds,
        learning_rate=learning_rate,
        epochs=epochs,
        optimizer_config=optimizer_config,
        loss_fn=loss_fn,
        scheduler_config=scheduler_config,
        callbacks=callbacks,
        enable_progress_bar=enable_progress_bar,
        log_every_n_steps=log_every_n_steps,
        unfreeze_epoch=unfreeze_epoch,
        enable_mlflow=enable_mlflow,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_run_name=mlflow_run_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )


def _train_single_fold(
    model: nn.Module,
    data_module,
    model_name: str = "encoder",
    learning_rate: float = 1e-3,
    epochs: int = 30,
    optimizer_config: Optional[dict] = None,
    loss_fn: str = "mse",
    scheduler_config: Optional[dict] = None,
    callbacks: Optional[list] = None,
    enable_progress_bar: bool = True,
    log_every_n_steps: int = 50,
    unfreeze_epoch: Optional[int] = None,
    enable_mlflow: bool = True,
    mlflow_experiment_name: str = "neural_encoder",
    mlflow_run_name: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
):
    """Train a single model (no cross-validation)."""

    # Create Lightning module
    lightning_model = EncoderLightningModule(
        model=model,
        learning_rate=learning_rate,
        optimizer_config=optimizer_config,
        loss_fn=loss_fn,
        scheduler_config=scheduler_config,
    )

    # Setup callbacks
    if callbacks is None:
        callbacks = []

    # Add default callbacks
    callbacks.extend([LearningRateMonitor(logging_interval="epoch")])

    # Add unfreeze callback if specified
    if unfreeze_epoch is not None:
        callbacks.append(UnfreezeCallback(unfreeze_epoch=unfreeze_epoch))

    # Add verification callback if MLflow is enabled
    if enable_mlflow:
        verification_callback = EncoderVerificationCallback(
            data_module=data_module,
            save_model=True,
            model_save_dir="workspace/models/encoders",
            plots_save_dir="workspace/plots/verification",
            enable_mlflow_logging=True,
        )
        callbacks.append(verification_callback)

    # Setup loggers
    loggers = []

    # MLflow logger if enabled
    if enable_mlflow:
        mlflow_logger = MLFlowLogger(
            experiment_name=mlflow_experiment_name,
            run_name=mlflow_run_name,
            tracking_uri=mlflow_tracking_uri,
            log_model=True,
        )
        loggers.append(mlflow_logger)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=enable_progress_bar,
        log_every_n_steps=log_every_n_steps,
        accelerator="cpu"
        if torch.backends.mps.is_available()
        else "auto",  # Force CPU on MPS to avoid compatibility issues
        devices=1 if torch.backends.mps.is_available() else "auto",
        deterministic=False,
        enable_checkpointing=False,
        # Disable to avoid MLflow artifact path issues
    )

    # Train the model
    trainer.fit(lightning_model, data_module)

    # Test the model
    trainer.test(lightning_model, data_module)

    # Log experiment to MLflow if enabled
    if enable_mlflow:
        # Prepare hyperparameters for logging
        hyperparams = {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "optimizer_config": optimizer_config or {},
            "loss_fn": loss_fn,
            "scheduler_config": scheduler_config or {},
            "model_name": model_name,
            "unfreeze_epoch": unfreeze_epoch,
        }

        # Prepare dataset info
        dataset_info = {
            "train_size": len(data_module.train_dataset),
            "val_size": len(data_module.val_dataset),
            "test_size": len(data_module.test_dataset),
            "batch_size": data_module.batch_size,
            "input_shape": data_module.images.shape,
            "output_neurons": data_module.firing_rates.shape[1],
        }

        # Save model path for logging - always save in encoders directory
        model_save_path = (
            f"workspace/models/encoders/{model_name}_"
            f"{mlflow_run_name or 'latest'}.pt"
        )

        # Log experiment
        log_encoder_experiment(
            model=model,
            lightning_module=lightning_model,
            hyperparams=hyperparams,
            dataset_info=dataset_info,
            model_save_path=model_save_path,
            experiment_name=mlflow_experiment_name,
            run_name=mlflow_run_name,
        )

    return trainer, lightning_model, data_module


def _train_with_cv(
    model: nn.Module,
    data_module,
    model_name: str = "encoder",
    n_folds: int = 5,
    **kwargs,
):
    """
    Train with k-fold cross-validation.

    Args:
        model: The model to train (will be cloned for each fold)
        data_module: Lightning data module
        model_name: Base name for the model
        n_folds: Number of cross-validation folds
        **kwargs: Additional arguments passed to _train_single_fold

    Returns:
        list: List of tuples (trainer, lightning_model, data_module)
        for each fold
    """
    from sklearn.model_selection import KFold

    # Get all training indices
    train_indices = np.arange(len(data_module.train_dataset))

    # Initialize k-fold splitter
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Store results for each fold
    fold_results = []

    print(f"Starting {n_folds}-fold cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_indices)):
        print(f"\n=== Fold {fold + 1}/{n_folds} ===")

        # Create fold-specific data module
        fold_data_module = data_module.__class__(
            train_indices=train_idx,
            val_indices=val_idx,
            **{
                k: v
                for k, v in data_module.__dict__.items()
                if not k.startswith("_")
                and k not in ["train_indices", "val_indices"]
            },
        )

        # Clone the model for this fold
        fold_model = type(model)(
            **{
                k: v
                for k, v in model.__dict__.items()
                if not k.startswith("_")
            }
        )
        fold_model.load_state_dict(model.state_dict())

        # Train the model for this fold
        fold_model_name = f"{model_name}_fold_{fold + 1}"

        try:
            result = _train_single_fold(
                model=fold_model,
                data_module=fold_data_module,
                model_name=fold_model_name,
                **kwargs,
            )

            fold_results.append(result)
            print(f"Fold {fold + 1} completed successfully")

        except Exception as e:
            print(f"Error in fold {fold + 1}: {e}")
            fold_results.append(None)

    # Print cross-validation summary
    successful_folds = [r for r in fold_results if r is not None]
    print("\n=== Cross-Validation Summary ===")
    print(f"Successful folds: {len(successful_folds)}/{n_folds}")

    if successful_folds:
        # Calculate average metrics across folds
        final_val_losses = []

        for _, lightning_model, _ in successful_folds:
            if lightning_model.val_losses:
                final_val_losses.append(lightning_model.val_losses[-1])

        if final_val_losses:
            avg_val_loss = np.mean(final_val_losses)
            std_val_loss = np.std(final_val_losses)
            print(
                "Average final validation loss: "
                f"{avg_val_loss:.4f} ± {std_val_loss:.4f}"
            )

    return fold_results


def train_simple_encoder(data_module, out_neurons: int, **kwargs):
    """
    Convenience function to train a SimpleEncoder.
    """
    model = SimpleEncoder(out_neurons=out_neurons)
    return train_encoder(
        model=model,
        data_module=data_module,
        model_name="simple_encoder",
        **kwargs,
    )


def train_skip_connection_encoder(data_module, out_neurons: int, **kwargs):
    """
    Convenience function to train a SimpleEncoderWithSkipConnection.
    """
    model = SimpleEncoderWithSkipConnection(out_neurons=out_neurons)
    return train_encoder(
        model=model,
        data_module=data_module,
        model_name="skip_connection_encoder",
        **kwargs,
    )


def train_resnet_encoder(
    data_module,
    out_neurons: int,
    resnet_type: str = "resnet18",
    freeze_backbone: bool = True,
    unfreeze_epoch: int = 15,
    **kwargs,
):
    """
    Convenience function to train a ResNetEncoder.
    """
    model = ResNetEncoder(
        out_neurons=out_neurons,
        resnet_type=resnet_type,
        freeze_backbone=freeze_backbone,
    )

    # Use differential learning rates for ResNet if not specified in kwargs
    if "optimizer_config" not in kwargs:
        kwargs["optimizer_config"] = {"type": "resnet_differential"}

    return train_encoder(
        model=model,
        data_module=data_module,
        model_name=f"resnet_{resnet_type}",
        unfreeze_epoch=unfreeze_epoch,
        **kwargs,
    )
