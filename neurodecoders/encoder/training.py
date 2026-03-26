"""
Training utilities for neural encoders.
"""

import os
import traceback
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import Callback
from sklearn.model_selection import KFold

from neurodecoders.core.base_lightning_module import BaseLightningModule
from neurodecoders.core.training_runner import (
    TrainingConfig,
    UnfreezeCallback,
    create_standard_callbacks,
    create_trainer,
)
from neurodecoders.encoder.verification_callback import (
    EncoderVerificationCallback,
)
from neurodecoders.mlflow_utils.utils import (
    log_cross_validation_metrics,
    log_dataset_input_and_params,
    log_training_config,
    log_training_metrics,
    setup_mlflow_experiment,
)
from neurodecoders.paths import get_path


class EncoderLightningModule(BaseLightningModule):
    """
    PyTorch Lightning module for training neural encoders.

    Inherits from BaseLightningModule and adds encoder-specific functionality:
    - Computing correlation metrics between predicted and true firing rates
    - Flexible optimizer_config/scheduler_config dict interface

    This module can work with any model architecture by accepting the model
    as a parameter instead of hardcoding it.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        optimizer_config: Optional[Dict[str, Any]] = None,
        loss_fn: str = "mse",
        scheduler_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the encoder lightning module.

        Args:
            model: The encoder model to train
            learning_rate: Learning rate for optimizer
            optimizer_config: Dict with 'type' (adam/adamw/sgd) and optional
                            'weight_decay', 'momentum' keys
            loss_fn: Loss function ('mse', 'l1', 'smooth_l1', 'huber')
            scheduler_config: Dict with 'type' (none/step/cosine/plateau) and
                            optional 'step_size', 'gamma', 'patience' keys
        """
        # Convert config format
        opt_config = optimizer_config or {"type": "adam"}
        sched_config = scheduler_config or {"type": "none"}

        super().__init__(
            model=model,
            learning_rate=learning_rate,
            optimizer_type=opt_config.get("type", "adam"),
            optimizer_config=opt_config,
            loss_fn=loss_fn,
            scheduler_type=sched_config.get("type", "none"),
            scheduler_config=sched_config,
        )

        # Store config for logging/debugging access
        self.optimizer_config = opt_config
        self.scheduler_config = sched_config

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Encoder training step: predict firing rates from images.

        Args:
            batch: Tuple of (images, firing_rates)
            batch_idx: Index of the batch

        Returns:
            Loss tensor
        """
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)

        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Encoder validation step.

        Args:
            batch: Tuple of (images, firing_rates)
            batch_idx: Index of the batch

        Returns:
            Loss tensor
        """
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Encoder test step with correlation metrics.

        Computes both loss and neuron-wise correlation between predicted
        and true firing rates.

        Args:
            batch: Tuple of (images, firing_rates)
            batch_idx: Index of the batch

        Returns:
            Loss tensor
        """
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True)

        # Calculate correlation coefficient between true and predicted firing rates
        y_np = y.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()

        correlations = []
        for i in range(y_np.shape[1]):
            corr = np.corrcoef(y_np[:, i], pred_np[:, i])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

        if correlations:
            avg_corr = np.mean(correlations)
            self.log(
                "test_correlation", avg_corr, on_step=False, on_epoch=True
            )

        return loss


def _train_single_model(
    model: nn.Module,
    data_module,
    model_name: str = "encoder",
    learning_rate: float = 1e-3,
    epochs: int = 100,
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

    # Build additional callbacks for encoder-specific needs
    additional_callbacks: List[Callback] = []
    if callbacks:
        additional_callbacks.extend(callbacks)

    # Add unfreeze callback if specified
    if unfreeze_epoch is not None:
        additional_callbacks.append(UnfreezeCallback(unfreeze_epoch=unfreeze_epoch))

    # Add verification callback if MLflow is enabled
    if enable_mlflow:
        verification_callback = EncoderVerificationCallback(
            data_module=data_module,
            save_model=True,
            model_save_dir=get_path("workspace/models/encoders"),
            enable_mlflow_logging=True,
        )
        additional_callbacks.append(verification_callback)

    # Create shared TrainingConfig
    config = TrainingConfig(
        epochs=epochs,
        batch_size=data_module.batch_size if hasattr(data_module, "batch_size") else 32,
        learning_rate=learning_rate,
        optimizer_type=optimizer_config.get("type", "adam") if optimizer_config else "adam",
        scheduler_type=scheduler_config.get("type", "none") if scheduler_config else "none",
        loss_fn=loss_fn,
        enable_mixed_precision=enable_mixed_precision,
        enable_early_stopping=enable_early_stopping,
        early_stopping_patience=early_stopping_patience,
        enable_checkpointing=enable_checkpointing,
        enable_mlflow=enable_mlflow,
        enable_progress_bar=enable_progress_bar,
        log_every_n_steps=log_every_n_steps,
        model_name=model_name,
    )

    # Create callbacks using shared infrastructure
    all_callbacks = create_standard_callbacks(
        config,
        checkpoint_filename_prefix=model_name,
        additional_callbacks=additional_callbacks,
    )

    # Find checkpoint_callback for later use (MLflow logging)
    checkpoint_callback = None
    for cb in all_callbacks:
        if hasattr(cb, "best_model_path"):
            checkpoint_callback = cb
            break

    # Setup loggers
    loggers: List[pl.loggers.Logger] = []

    # Add MLflow logger if enabled
    if enable_mlflow:
        # Get tracking URI from config (supports both database and file system)
        from neurodecoders.config import get_mlflow_tracking_uri
        tracking_uri = get_mlflow_tracking_uri()
        
        # If using file system, ensure directory exists
        if tracking_uri and tracking_uri.startswith("file:"):
            configured_mlflow_dir = tracking_uri.replace("file://", "").replace("file:", "")
            try:
                os.makedirs(configured_mlflow_dir, exist_ok=True)
            except OSError as e:
                print(
                    f"Warning: Could not create MLflow directory "
                    f"{configured_mlflow_dir}: {e}"
                )
            
            # Check directory accessibility
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
        
        # Set up MLflow with proper artifact location
        from neurodecoders.config import get_base_path
        artifact_location = f"file://{get_base_path()}/mlruns"
        
        setup_mlflow_experiment(
            experiment_name=mlflow_experiment_name, 
            tracking_uri=tracking_uri,
            artifact_location=artifact_location
        )
        print(f"Using MLflow tracking URI: {tracking_uri}")
        print(f"Using MLflow artifact location: {artifact_location}")

        print(f"Setting MLflow experiment: {mlflow_experiment_name}")

        if mlflow_run_name:
            mlflow.start_run(run_name=mlflow_run_name, log_system_metrics=True)
        else:
            mlflow.start_run(log_system_metrics=True)

        # Log training parameters via shared util
        try:
            training_params = {
                "learning_rate": learning_rate,
                "epochs": epochs,
                "model_type": model_name,
                "loss_function": loss_fn,
                "optimizer_type": (
                    lightning_model.optimizer_config.get("type", "adam")
                ),
                "scheduler_type": (
                    lightning_model.scheduler_config.get("type", "none")
                ),
                "batch_size": (
                    data_module.batch_size
                    if hasattr(data_module, "batch_size")
                    else "unknown"
                ),
                "enable_mixed_precision": enable_mixed_precision,
                "enable_early_stopping": enable_early_stopping,
                "early_stopping_patience": early_stopping_patience,
            }
            log_training_config(training_params)
            print("Logged training parameters to MLflow")
        except Exception as e:
            print(f"Warning: Could not log training parameters to MLflow: {e}")
            traceback.print_exc()

        # Log dataset metadata via shared util
        try:
            log_dataset_input_and_params(data_module)
            print("Logged neural dataset metadata to MLflow")
        except Exception as e:
            print(f"Warning: Could not log dataset metadata to MLflow: {e}")
            traceback.print_exc()

    # Create trainer using shared infrastructure
    trainer = create_trainer(config, callbacks=all_callbacks, loggers=loggers)

    # Train the model
    trainer.fit(lightning_model, data_module)

    # Test the model
    test_results = trainer.test(lightning_model, data_module)

    # Log final metrics to MLflow if enabled
    if enable_mlflow:
        try:
            # Log final test loss if available
            if test_results:
                test_loss = test_results[0].get("test_loss")
                if test_loss is not None:
                    log_training_metrics(
                        {"test_loss": float(test_loss)}, step=epochs
                    )

            # Log checkpoint path as parameter (not metric)
            if enable_checkpointing and checkpoint_callback and checkpoint_callback.best_model_path:
                mlflow.log_param(
                    "best_checkpoint_path", checkpoint_callback.best_model_path
                )

            # Log the trained model
            try:
                from neurodecoders.mlflow_utils.utils import (
                    log_model_artifacts,
                )

                # Create dataset info for model logging
                dataset_info = {
                    "n_images": data_module.n_images,
                    "n_neurons": data_module.n_neurons,
                    "image_shape": str(data_module.images.shape[1:]),
                    "dataset_type": getattr(
                        data_module, "dataset_type", "unknown"
                    ),
                    "sta_pattern": getattr(
                        data_module, "sta_pattern", "unknown"
                    ),
                }

                # Create training info for model logging
                model_training_info = {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "batch_size": data_module.batch_size,
                    "train_loss": lightning_model.train_losses[-1]
                    if lightning_model.train_losses
                    else None,
                    "val_loss": lightning_model.val_losses[-1]
                    if lightning_model.val_losses
                    else None,
                    "test_loss": test_results[0].get("test_loss")
                    if test_results
                    else None,
                }

                log_model_artifacts(
                    model=lightning_model,
                    model_name="encoder_model",
                    model_type="encoder",
                    dataset_info=dataset_info,
                    training_info=model_training_info,
                )
                print("Logged encoder model to MLflow")

            except Exception as e:
                print(f"Warning: Could not log model to MLflow: {e}")
                traceback.print_exc()

        except Exception as e:
            print(f"Warning: Could not log final metrics to MLflow: {e}")
            traceback.print_exc()

    # End the MLflow run AFTER all callbacks have finished
    if enable_mlflow:
        mlflow.end_run()

    return trainer, lightning_model, data_module


def _clone_model(model: nn.Module) -> nn.Module:
    """
    Create a deep copy of a model with fresh weights.

    Args:
        model: The model to clone

    Returns:
        A new model instance with the same architecture but fresh weights
    """
    # Get the model class
    model_class = model.__class__

    # Extract parameters from model architecture
    if model_class.__name__ == "Simple3LayerEncoder":
        # Extract parameters for Simple3LayerEncoder
        out_neurons = model.out_neurons
        image_height = model.image_height
        image_width = model.image_width
        learn_positions = model.learn_positions
        cloned_model = model_class(
            out_neurons=out_neurons,
            image_height=image_height,
            image_width=image_width,
            learn_positions=learn_positions,
        )
    elif (
        model_class.__name__ == "SimpleEncoder"
        or model_class.__name__ == "SimpleEncoderWithSkipConnection"
    ):
        # Extract out_neurons from the last linear layer of the fc module
        # Find the last Linear layer (before the ELU activation)
        for layer in reversed(model.fc):
            if hasattr(layer, "out_features"):
                out_neurons = layer.out_features
                break
        else:
            raise ValueError("Could not find Linear layer with out_features")
        cloned_model = model_class(out_neurons=out_neurons)

    elif model_class.__name__ == "ResNetEncoder":
        # Extract out_neurons from the last layer of the firing_head module
        out_neurons = model.firing_head[-1].out_features
        freeze_backbone = getattr(model, "freeze_backbone", True)
        cloned_model = model_class(
            out_neurons=out_neurons,
            freeze_backbone=freeze_backbone,
        )
    elif model_class.__name__ == "ResNetFromScratch":
        # Extract out_neurons from the last layer of the firing_head module
        out_neurons = model.firing_head[-1].out_features
        freeze_backbone = getattr(model, "freeze_backbone", False)
        cloned_model = model_class(
            out_neurons=out_neurons,
            freeze_backbone=freeze_backbone,
        )
    elif model_class.__name__ == "ResNetConvOnly":
        # firing_head is a single Linear layer, not Sequential
        out_neurons = model.firing_head.out_features
        freeze_backbone = getattr(model, "freeze_backbone", True)
        cloned_model = model_class(
            out_neurons=out_neurons,
            freeze_backbone=freeze_backbone,
        )
    elif model_class.__name__ == "ResNetConv_2layerHead":
        # firing_head is a Sequential with 2 linear layers
        out_neurons = model.firing_head[-1].out_features
        freeze_backbone = getattr(model, "freeze_backbone", True)
        cloned_model = model_class(
            out_neurons=out_neurons,
            freeze_backbone=freeze_backbone,
        )
    else:
        # Fallback: try to recreate with default parameters
        try:
            cloned_model = model_class()
        except Exception as e:
            raise ValueError(f"Cannot clone model {model_class.__name__}: {e}")

    return cloned_model


def train_encoder(
    model: nn.Module,
    data_module,
    model_name: str = "encoder",
    n_folds: int = 5,
    learning_rate: float = 1e-3,
    epochs: int = 100,
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
    early_stopping_patience: int = 100,
    enable_checkpointing: bool = True,
):
    """
    Generic training function that works with any model architecture.
    Supports k-fold cross-validation with proper model isolation.

    Args:
        model: The neural network model to train (will be cloned for each fold)
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
        unfreeze_epoch: Epoch to start unfreezing backbone
        (for transfer learning)
        enable_mlflow: Whether to enable MLflow logging
        mlflow_experiment_name: MLflow experiment name
        mlflow_run_name: MLflow run name
        mlflow_tracking_uri: MLflow tracking URI
        n_folds: Number of cross-validation folds
        enable_mixed_precision: Whether to use mixed precision training
        enable_early_stopping: Whether to enable early stopping
        early_stopping_patience: Patience for early stopping
        enable_checkpointing: Whether to enable model checkpointing

    Returns:
        list of (trainer, lightning_model, data_module) tuples for all folds
    """

    # Handle single fold case (no cross-validation)
    if n_folds == 1:
        print("\n=== Single Training Run (No Cross-Validation) ===")

        # Setup MLflow for single run
        if enable_mlflow:
            from neurodecoders.config import get_mlflow_tracking_uri, get_base_path
            tracking_uri = get_mlflow_tracking_uri()
            artifact_location = f"file://{get_base_path()}/mlruns"
            setup_mlflow_experiment(
                experiment_name=mlflow_experiment_name,
                tracking_uri=tracking_uri,
                artifact_location=artifact_location,
            )

        # Use the data module as-is (with existing train/val splits)
        fold_model = _clone_model(model)
        fold_data_module = data_module

        # Train single model
        trainer, lightning_model, _ = _train_single_model(
            model=fold_model,
            data_module=fold_data_module,
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
            enable_mixed_precision=enable_mixed_precision,
            enable_early_stopping=enable_early_stopping,
            early_stopping_patience=early_stopping_patience,
            enable_checkpointing=enable_checkpointing,
        )

        return [(trainer, lightning_model, fold_data_module)]

    # Multi-fold cross-validation
    print(f"\n=== K-Fold Cross-Validation ({n_folds} folds) ===")

    # Setup MLflow for cross-validation
    if enable_mlflow:
        from neurodecoders.config import get_mlflow_tracking_uri, get_base_path
        tracking_uri = get_mlflow_tracking_uri()
        artifact_location = f"file://{get_base_path()}/mlruns"
        setup_mlflow_experiment(
            experiment_name=mlflow_experiment_name, 
            tracking_uri=tracking_uri,
            artifact_location=artifact_location,
        )

        # Start main CV run to log aggregated results
        cv_run_name = (
            f"{mlflow_run_name}_cv_summary"
            if mlflow_run_name
            else "cv_summary"
        )

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_indices = np.arange(len(data_module.full_dataset))

    fold_results = []
    fold_metrics: dict[str, list[float]] = {
        "train_losses": [],
        "val_losses": [],
        "test_losses": [],
        "test_correlations": [],
    }

    for fold in range(n_folds):
        print(f"\n=== Training Fold {fold + 1}/{n_folds} ===")

        # Create a fresh model instance for this fold
        fold_model = _clone_model(model)

        # Create fold-specific data module
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

        # Train this fold with its own model instance
        trainer, lightning_model, _ = _train_single_model(
            model=fold_model,  # Use the cloned model
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

        fold_results.append((trainer, lightning_model, fold_data_module))

        # Collect metrics from this fold
        if (
            hasattr(lightning_model, "train_losses")
            and lightning_model.train_losses
        ):
            fold_metrics["train_losses"].append(
                lightning_model.train_losses[-1]
            )
        if (
            hasattr(lightning_model, "val_losses")
            and lightning_model.val_losses
        ):
            fold_metrics["val_losses"].append(lightning_model.val_losses[-1])

        # Get test metrics from trainer (single final test per fold)
        test_results = trainer.test(
            lightning_model, fold_data_module, verbose=False
        )
        if test_results:
            test_metrics = test_results[0]
            if "test_loss" in test_metrics:
                fold_metrics["test_losses"].append(test_metrics["test_loss"])
            if "test_correlation" in test_metrics:
                fold_metrics["test_correlations"].append(test_metrics["test_correlation"]) 

    # Log aggregated cross-validation results
    if enable_mlflow:
        with mlflow.start_run(run_name=cv_run_name, nested=True):
            # Log CV parameters
            log_training_config(
                {
                    "cv_folds": n_folds,
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "model_type": model_name,
                    "loss_function": loss_fn,
                }
            )

            # Calculate and log aggregated metrics via shared util
            cv_results = {}
            if fold_metrics["train_losses"]:
                cv_results["train_losses"] = fold_metrics["train_losses"]
            if fold_metrics["val_losses"]:
                cv_results["val_losses"] = fold_metrics["val_losses"]
            if fold_metrics["test_losses"]:
                cv_results["test_losses"] = fold_metrics["test_losses"]
            if fold_metrics["test_correlations"]:
                cv_results["test_correlations"] = fold_metrics[
                    "test_correlations"
                ]

            if cv_results:
                log_cross_validation_metrics(cv_results)

            # Log individual fold results
            for fold in range(n_folds):
                if fold < len(fold_metrics["train_losses"]):
                    mlflow.log_metric(
                        f"fold_{fold + 1}_train_loss",
                        fold_metrics["train_losses"][fold],
                    )
                if fold < len(fold_metrics["val_losses"]):
                    mlflow.log_metric(
                        f"fold_{fold + 1}_val_loss",
                        fold_metrics["val_losses"][fold],
                    )
                if fold < len(fold_metrics["test_losses"]):
                    mlflow.log_metric(
                        f"fold_{fold + 1}_test_loss",
                        fold_metrics["test_losses"][fold],
                    )
                if fold < len(fold_metrics["test_correlations"]):
                    mlflow.log_metric(
                        f"fold_{fold + 1}_test_correlation",
                        fold_metrics["test_correlations"][fold],
                    )

            print("\n=== Cross-Validation Summary ===")
            if fold_metrics["test_losses"]:
                print(
                    f"Average Test Loss: "
                    f"{np.mean(fold_metrics['test_losses']):.4f} ± "
                    f"{np.std(fold_metrics['test_losses']):.4f}"
                )
            if fold_metrics["test_correlations"]:
                print(
                    f"Average Test Correlation: "
                    f"{np.mean(fold_metrics['test_correlations']):.4f} ± "
                    f"{np.std(fold_metrics['test_correlations']):.4f}"
                )
            print(f"Results logged to MLflow run: {cv_run_name}")

    return fold_results
