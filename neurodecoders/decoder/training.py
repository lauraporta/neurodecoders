"""
Training utilities for neural decoders.
"""

from typing import List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from neurodecoders.data import NeuralDataModule
from neurodecoders.decoder.models import SimpleDecoder
from neurodecoders.mlflow_utils.utils import (
    log_training_config,
    log_training_metrics,
    setup_mlflow_experiment,
)
from neurodecoders.paths import get_path


class DecoderLightningModule(pl.LightningModule):
    """Generic Lightning module for decoder training."""

    def __init__(
        self,
        in_neurons: int,
        image_size: int = 64,
        learning_rate: float = 1e-4,
        loss_fn: str = "mse",
        optimizer_type: str = "adam",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = SimpleDecoder(in_neurons, image_size)
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, firing_rates = batch
        pred = self(firing_rates)
        loss = self.criterion(pred, images)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, firing_rates = batch
        pred = self(firing_rates)
        loss = self.criterion(pred, images)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, firing_rates = batch
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
            log_training_metrics(
                {"train_loss": float(loss)}, step=self.current_epoch
            )

    def on_validation_epoch_end(self):
        loss = self.trainer.callback_metrics.get("val_loss")
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        if loss is not None:
            self.val_losses.append(loss)
            log_training_metrics(
                {"val_loss": float(loss)}, step=self.current_epoch
            )

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
    )

    callbacks: List[pl.Callback] = [
        LearningRateMonitor(logging_interval="epoch")
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

    setup_mlflow_experiment(experiment_name=mlflow_experiment_name)
    if mlflow_run_name:
        import mlflow

        mlflow.start_run(run_name=mlflow_run_name, log_system_metrics=True)
        log_training_config(
            {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "optimizer_type": optimizer,
                "loss_function": loss_fn,
            }
        )

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
