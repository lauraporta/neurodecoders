"""
Generic training pipeline for neural encoder models.

This module provides a unified training interface that works with any
model architecture from the models module.
"""

from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger

from .mlflow_utils import log_encoder_experiment
from .verification_callback import create_verification_callback


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
        weight_decay: float = 1e-5,
        optimizer_config: Optional[dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.loss_fn = nn.MSELoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_config = optimizer_config or {"type": "adam"}

        # Store training history for plotting
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)

        # Log training loss
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)

        # Log validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)

        # Log test loss
        self.log("test_loss", loss, on_step=False, on_epoch=True)
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
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": backbone_params,
                    "lr": self.learning_rate * 0.1,  # Lower LR for backbone
                    "weight_decay": self.weight_decay,
                },
            ]

            optimizer = torch.optim.Adam(param_groups)

        else:
            # Standard optimizer for all parameters
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

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
    weight_decay: float = 1e-5,
    epochs: int = 30,
    optimizer_config: Optional[dict] = None,
    callbacks: Optional[list] = None,
    enable_progress_bar: bool = True,
    log_every_n_steps: int = 50,
    logger_name: str = "encoder",
    unfreeze_epoch: Optional[int] = None,
    enable_mlflow: bool = True,
    mlflow_experiment_name: str = "neural_encoder",
    mlflow_run_name: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
):
    """
    Generic training function that works with any model architecture.

    Args:
        model: The neural network model to train
        data_module: Lightning data module with train/val/test dataloaders
        model_name: Name for the model (used in logging)
        learning_rate: Learning rate for training
        weight_decay: Weight decay for regularization
        epochs: Number of training epochs
        optimizer_config: Dictionary specifying optimizer configuration
        callbacks: List of additional callbacks
        enable_progress_bar: Whether to show progress bar
        log_every_n_steps: Logging frequency
        logger_name: Name for the experiment logger
        unfreeze_epoch: Epoch to start unfreezing backbone (for transfer
        learning)

    Returns:
        trainer: The trained trainer object
        lightning_model: The trained Lightning module
        data_module: The data module
    """

    # Create Lightning module
    lightning_model = EncoderLightningModule(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_config=optimizer_config,
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
        verification_callback = create_verification_callback(
            data_module=data_module,
            save_model=True,
            model_save_dir="workspace/models/encoders",
            plots_save_dir="workspace/plots/verification",
            enable_mlflow_logging=True,
        )
        callbacks.append(verification_callback)

    # Setup loggers
    loggers = []

    # TensorBoard logger
    tensorboard_logger = TensorBoardLogger(
        "workspace/logs/lightning_logs", name=logger_name
    )
    loggers.append(tensorboard_logger)

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
            "weight_decay": weight_decay,
            "epochs": epochs,
            "optimizer_config": optimizer_config or {},
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

        # Save model path for logging
        model_save_path = (
            f"workspace/models/{model_name}_{mlflow_run_name or 'latest'}.pt"
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


def train_simple_encoder(data_module, out_neurons: int, **kwargs):
    """
    Convenience function to train a SimpleEncoder.
    """
    from .models import SimpleEncoder

    model = SimpleEncoder(out_neurons=out_neurons)
    return train_encoder(
        model=model,
        data_module=data_module,
        model_name="simple_encoder",
        logger_name="simple_encoder",
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
    from .models import ResNetEncoder

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
        logger_name=f"resnet_{resnet_type}",
        unfreeze_epoch=unfreeze_epoch,
        **kwargs,
    )
