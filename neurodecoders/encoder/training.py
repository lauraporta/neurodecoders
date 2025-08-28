"""
Training utilities for neural encoders.
"""

import os
import traceback
from typing import List, Optional

import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from sklearn.model_selection import KFold

from neurodecoders.encoder.verification_callback import (
    EncoderVerificationCallback,
)
from neurodecoders.paths import get_mlflow_path, get_path


class MLflowHistoryCallback(Callback):
    """
    Callback to log training and validation history to MLflow for every epoch.

    This callback captures epoch-level metrics and logs them to MLflow with
    the epoch number as the step, enabling proper history tracking.
    """

    def __init__(self):
        super().__init__()
        self.current_epoch = 0

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """Log training metrics to MLflow at the end of each training epoch."""
        self.current_epoch = trainer.current_epoch

        # Get training loss from logged metrics
        if trainer.logged_metrics:
            train_loss = trainer.logged_metrics.get("train_loss")
            if train_loss is not None:
                # Convert tensor to float if needed
                if isinstance(train_loss, torch.Tensor):
                    train_loss = train_loss.item()

                # Log to MLflow with epoch as step
                mlflow.log_metric(
                    "train_loss", train_loss, step=self.current_epoch
                )

                # Also store in the module's history list
                if not hasattr(pl_module, "train_losses"):
                    pl_module.train_losses = []
                pl_module.train_losses.append(train_loss)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """
        Log validation metrics to MLflow at the end of each validation epoch.
        """
        self.current_epoch = trainer.current_epoch

        # Get validation loss from logged metrics
        if trainer.logged_metrics:
            val_loss = trainer.logged_metrics.get("val_loss")
            if val_loss is not None:
                # Convert tensor to float if needed
                if isinstance(val_loss, torch.Tensor):
                    val_loss = val_loss.item()

                # Log to MLflow with epoch as step
                mlflow.log_metric(
                    "val_loss", val_loss, step=self.current_epoch
                )

                # Also store in the module's history list
                if not hasattr(pl_module, "val_losses"):
                    pl_module.val_losses = []
                pl_module.val_losses.append(val_loss)


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

        # Calculate correlation for each neuron
        correlations = []
        for i in range(y_np.shape[1]):
            corr = np.corrcoef(y_np[:, i], pred_np[:, i])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

        # Log average correlation
        if correlations:
            avg_corr = np.mean(correlations)
            self.log(
                "test_correlation", avg_corr, on_step=False, on_epoch=True
            )

        return loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer_type = self.optimizer_config.get("type", "adam")
        weight_decay = self.optimizer_config.get("weight_decay", 0.0)

        if optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        # Configure scheduler
        scheduler_type = self.scheduler_config.get("type", "none")
        if scheduler_type == "none":
            return optimizer
        elif scheduler_type == "step":
            step_size = self.scheduler_config.get("step_size", 30)
            gamma = self.scheduler_config.get("gamma", 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs
            )
        elif scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=5,
                verbose=True,
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss" if scheduler_type == "plateau" else None,
            },
        }


class UnfreezeCallback(pl.Callback):
    """Callback to unfreeze backbone layers at a specific epoch."""

    def __init__(self, unfreeze_epoch: int):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.unfreeze_epoch:
            print(f"Unfreezing backbone at epoch {self.unfreeze_epoch}")
            for param in pl_module.model.backbone.parameters():
                param.requires_grad = True


def _train_single_model(
    model: nn.Module,
    data_module,
    model_name: str = "encoder",
    learning_rate: float = 1e-3,
    epochs: int = 2,
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
    enable_mixed_precision: bool = True,
    enable_early_stopping: bool = True,
    early_stopping_patience: int = 100,
    enable_checkpointing: bool = True,
):
    """Train a single model instance."""

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

    # Add early stopping if enabled
    if enable_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                mode="min",
                verbose=True,
            )
        )

    # Add model checkpointing if enabled
    if enable_checkpointing:
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=get_path("workspace/checkpoints"),
            filename=f"{model_name}-{{epoch:02d}}-{{val_loss:.4f}}",
            save_top_k=3,
            mode="min",
            verbose=True,
        )
        callbacks.append(checkpoint_callback)

    # Add unfreeze callback if specified
    if unfreeze_epoch is not None:
        callbacks.append(UnfreezeCallback(unfreeze_epoch=unfreeze_epoch))

    # Add MLflow history callback if MLflow is enabled
    if enable_mlflow:
        callbacks.append(MLflowHistoryCallback())

        # Add verification callback
        verification_callback = EncoderVerificationCallback(
            data_module=data_module,
            save_model=True,
            model_save_dir=get_path("workspace/models/encoders"),
            plots_save_dir=get_path("workspace/plots/verification"),
            enable_mlflow_logging=True,
        )
        callbacks.append(verification_callback)

    # Setup loggers
    loggers: List[pl.loggers.Logger] = []

    # Add MLflow logger if enabled
    if enable_mlflow:
        # Force MLflow tracking to the configured base path
        configured_mlflow_dir = get_mlflow_path()
        try:
            os.makedirs(configured_mlflow_dir, exist_ok=True)
        except OSError as e:
            print(
                f"Warning: Could not create MLflow directory "
                f"{configured_mlflow_dir}: {e}"
            )
        tracking_uri = f"file:{configured_mlflow_dir}"
        mlflow.set_tracking_uri(tracking_uri)
        print(f"Using MLflow tracking URI from config: {tracking_uri}")

        # Set experiment
        print(f"Setting MLflow experiment: {mlflow_experiment_name}")

        # Inform user about directory status
        if os.path.exists(configured_mlflow_dir):
            if os.access(configured_mlflow_dir, os.W_OK):
                print(
                    f"MLflow tracking directory is ready: "
                    f"{configured_mlflow_dir}"
                )
            else:
                print(
                    f"Warning: MLflow tracking directory not writable: "
                    f"{configured_mlflow_dir}"
                )
        else:
            print(
                f"Warning: MLflow tracking directory still does not exist: "
                f"{configured_mlflow_dir}"
            )

        mlflow.set_experiment(mlflow_experiment_name)

        # Start MLflow run with GPU monitoring
        if mlflow_run_name:
            mlflow.start_run(run_name=mlflow_run_name, log_system_metrics=True)
        else:
            mlflow.start_run(log_system_metrics=True)

        # Log dataset metadata
        try:
            # Create metadata summary DataFrame
            metadata_summary = pd.DataFrame(
                [
                    {
                        "dataset_id": data_module.get_dataset_id(),
                        "git_commit": data_module.git_commit,
                        "git_branch": data_module.git_branch,
                        "timestamp": data_module.timestamp,
                        "images_shape": str(data_module.images.shape),
                        "firing_rates_shape": str(
                            data_module.firing_rates.shape
                        ),
                        "labels_shape": str(data_module.labels.shape)
                        if data_module.labels is not None
                        else "None",
                        "total_size_mb": data_module.total_size_mb,
                        "dataset_type": data_module.dataset_type,
                        "sta_pattern": data_module.sta_pattern,
                        "n_neurons": data_module.n_neurons,
                        "n_images": data_module.n_images,
                    }
                ]
            )

            # Create source information
            source_info = (
                f"{get_path('workspace/datasets/synthetic')}/"
                f"{data_module.dataset_filename or 'unknown'}"
            )

            # Log the metadata dataset
            dataset_id = data_module.get_dataset_id()
            data_module.get_metadata_summary()
            metadata_name = f"neural_data_{dataset_id}"

            summary_dataset = mlflow.data.from_pandas(
                metadata_summary,
                source=source_info,
                name=metadata_name,
            )

            mlflow.log_input(summary_dataset, context="training_data")

            # Also log as parameters for backward compatibility
            dataset_params = data_module.get_mlflow_parameters()
            mlflow.log_params(dataset_params)

            print("Logged neural dataset metadata to MLflow:")
            print(f"  Dataset ID: {dataset_id}")
            print(f"  Git Commit: {data_module.git_commit}")
            print(f"  Git Branch: {data_module.git_branch}")
            print(f"  Training Timestamp: {data_module.timestamp}")
            print(f"  Images: {data_module.images.shape}")
            print(f"  Firing Rates: {data_module.firing_rates.shape}")
            labels_shape = (
                data_module.labels.shape
                if data_module.labels is not None
                else "None"
            )
            print(f"  Labels: {labels_shape}")
            print(f"  Total Size: {data_module.total_size_mb:.2f} MB")

        except Exception as e:
            print(f"Warning: Could not log dataset metadata to MLflow: {e}")
            traceback.print_exc()

    # Add LearningRateMonitor if we have loggers
    if loggers:
        callbacks.extend([LearningRateMonitor(logging_interval="epoch")])

    # Create trainer with enhanced configuration
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=enable_progress_bar,
        log_every_n_steps=log_every_n_steps,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        deterministic=False,  # set to true for reproducibility (pseudorandom)
        enable_checkpointing=enable_checkpointing,
        precision="16-mixed" if enable_mixed_precision else "32",
    )

    # Train the model
    trainer.fit(lightning_model, data_module)

    # Test the model
    trainer.test(lightning_model, data_module)

    # Log final metrics to MLflow if enabled
    if enable_mlflow:
        try:
            # Log final training info
            training_info = {
                "final_train_loss": lightning_model.train_losses[-1]
                if lightning_model.train_losses
                else None,
                "final_val_loss": lightning_model.val_losses[-1]
                if lightning_model.val_losses
                else None,
            }

            # Log metrics (only numeric values)
            mlflow.log_metrics(training_info)

            # Log checkpoint path as parameter (not metric)
            if enable_checkpointing and checkpoint_callback.best_model_path:
                mlflow.log_param(
                    "best_checkpoint_path", checkpoint_callback.best_model_path
                )

        except Exception as e:
            print(f"Warning: Could not log final metrics to MLflow: {e}")
            traceback.print_exc()

    # End the MLflow run AFTER all callbacks have finished
    if enable_mlflow:
        mlflow.end_run()

    return trainer, lightning_model, data_module


def train_encoder(
    model: nn.Module,
    data_module,
    model_name: str = "encoder",
    n_folds: int = 5,
    learning_rate: float = 1e-3,
    epochs: int = 2,  # Set to 2 epochs for testing
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
    enable_mixed_precision: bool = True,
    enable_early_stopping: bool = True,
    early_stopping_patience: int = 100,  # Updated default
    enable_checkpointing: bool = True,
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
        n_folds: Number of cross-validation folds (1 = single training)
        enable_mixed_precision: Whether to use mixed precision training
        enable_early_stopping: Whether to enable early stopping
        early_stopping_patience: Patience for early stopping
        enable_checkpointing: Whether to enable model checkpointing

    Returns:
        If n_folds=1: (trainer, lightning_model, data_module)
        If n_folds>1: list of (trainer, lightning_model, data_module) tuples
    """

    results = []

    if n_folds == 1:
        # Single training - use original data module
        print("\n=== Single Training ===")
        fold_data_module = data_module
        fold_run_name = mlflow_run_name or model_name
        fold_model_name = model_name

        # Train single model
        trainer, lightning_model, _ = _train_single_model(
            model=model,
            data_module=fold_data_module,
            model_name=fold_model_name,
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
            mlflow_run_name=fold_run_name,
            enable_mixed_precision=enable_mixed_precision,
            enable_early_stopping=enable_early_stopping,
            early_stopping_patience=early_stopping_patience,
            enable_checkpointing=enable_checkpointing,
        )

        return (trainer, lightning_model, fold_data_module)
    else:
        # Cross-validation - create k-fold splits
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        all_indices = np.arange(len(data_module.full_dataset))

        for fold in range(n_folds):
            print(f"\n=== Training Fold {fold + 1}/{n_folds} ===")

            # Cross-validation - create fold-specific data module
            train_indices, val_indices = list(kfold.split(all_indices))[fold]

            fold_data_module = data_module.__class__(
                images=data_module.images,
                firing_rates=data_module.firing_rates,
                labels=data_module.labels,
                batch_size=data_module.batch_size,
                dataset_metadata=data_module.dataset_metadata,
                use_memory_mapping=data_module.use_memory_mapping,
                chunk_size=data_module.chunk_size,
                prefetch_factor=data_module.prefetch_factor,
                num_workers=data_module.num_workers,
                pin_memory=data_module.pin_memory,
            )

            # Set custom splits for this fold
            fold_data_module.train_indices = train_indices
            fold_data_module.val_indices = val_indices
            fold_data_module.setup_splits()

            # Create fold-specific run name
            fold_run_name = (
                f"{mlflow_run_name}_fold_{fold + 1}"
                if mlflow_run_name
                else f"fold_{fold + 1}"
            )
            fold_model_name = f"{model_name}_fold_{fold + 1}"

            # Train this fold
            trainer, lightning_model, _ = _train_single_model(
                model=model,
                data_module=fold_data_module,
                model_name=fold_model_name,
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
                mlflow_run_name=fold_run_name,
                enable_mixed_precision=enable_mixed_precision,
                enable_early_stopping=enable_early_stopping,
                early_stopping_patience=early_stopping_patience,
                enable_checkpointing=enable_checkpointing,
            )

            results.append((trainer, lightning_model, fold_data_module))

        return results
