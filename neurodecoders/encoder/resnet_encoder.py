import datetime
import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

warnings.filterwarnings("ignore")


class ResNetEncoder(nn.Module):
    """
    Neural encoder using ResNet as backbone for predicting firing rates from
    images.
    Uses transfer learning with a pre-trained ResNet model.
    """

    def __init__(
        self, out_neurons, resnet_type="resnet18", freeze_backbone=True
    ):
        super().__init__()

        # Load pre-trained ResNet
        if resnet_type == "resnet18":
            self.backbone = models.resnet18(pretrained=True)
            feature_dim = 512
        elif resnet_type == "resnet34":
            self.backbone = models.resnet34(pretrained=True)
            feature_dim = 512
        elif resnet_type == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")

        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Add firing rate prediction head
        self.firing_head = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(512, out_neurons),
        )

        self.resnet_type = resnet_type
        self.feature_dim = feature_dim

    def forward(self, x):
        # ResNet expects 3 channels, ensure input is correct
        if x.shape[1] == 1:  # If grayscale, repeat to 3 channels
            x = x.repeat(1, 3, 1, 1)

        # Extract features from backbone
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten

        # Predict firing rates
        firing_rates = self.firing_head(features)

        return firing_rates

    def unfreeze_backbone(self, num_layers=None):
        """
        Unfreeze backbone layers for fine-tuning.
        Args:
            num_layers: Number of layers to unfreeze from the end (None = all
            layers)
        """
        backbone_layers = list(self.backbone.children())

        if num_layers is None:
            # Unfreeze all layers
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Unfreeze last num_layers
            layers_to_unfreeze = backbone_layers[-num_layers:]
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True

        print(f"Unfroze {num_layers if num_layers else 'all'} backbone layers")


class ResNetEncoderLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for ResNet-based encoder training.
    """

    def __init__(
        self,
        out_neurons: int,
        resnet_type="resnet18",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = ResNetEncoder(out_neurons, resnet_type, freeze_backbone)
        self.loss_fn = nn.MSELoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Store losses for plotting
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, firing_rates = batch
        predicted_rates = self(images)
        loss = self.loss_fn(predicted_rates, firing_rates)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, firing_rates = batch
        predicted_rates = self(images)
        loss = self.loss_fn(predicted_rates, firing_rates)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, firing_rates = batch
        predicted_rates = self(images)
        loss = self.loss_fn(predicted_rates, firing_rates)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # Use different learning rates for backbone and head
        backbone_params = list(self.model.backbone.parameters())
        head_params = list(self.model.firing_head.parameters())

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": backbone_params,
                    "lr": self.learning_rate * 0.1,
                },  # Lower LR for backbone
                {"params": head_params, "lr": self.learning_rate},
            ],
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

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


def train_resnet_encoder(
    images,
    firing_rates,
    resnet_type="resnet18",
    train_split=0.7,
    val_split=0.15,
    batch_size=32,
    learning_rate=1e-3,
    epochs=30,
    freeze_backbone=True,
    unfreeze_epoch=15,  # Epoch to start unfreezing backbone
    enable_progress_bar=True,
    log_every_n_steps=50,
    callbacks=None,
):
    """
    Train ResNet-based encoder using PyTorch Lightning

    Args:
        images: Input images
        firing_rates: Target firing rates
        resnet_type: Type of ResNet ('resnet18', 'resnet34', 'resnet50')
        freeze_backbone: Whether to freeze backbone initially
        unfreeze_epoch: Epoch to start unfreezing backbone layers
        ... (other args same as before)

    Returns:
        trainer: The trained trainer object
        model: The trained model
        data_module: The data module
    """
    from neurodecoders.encoder.utils import (
        NeuralDataModule,  # Import from utils module
    )

    # Create data module
    data_module = NeuralDataModule(
        images=images,
        firing_rates=firing_rates,
        train_split=train_split,
        val_split=val_split,
        batch_size=batch_size,
    )

    # Create model
    model = ResNetEncoderLightningModule(
        out_neurons=firing_rates.shape[1],
        resnet_type=resnet_type,
        learning_rate=learning_rate,
        freeze_backbone=freeze_backbone,
    )

    # Setup callbacks
    if callbacks is None:
        callbacks = []

    # Add default callbacks
    callbacks.extend([LearningRateMonitor(logging_interval="epoch")])

    # Add callback to unfreeze backbone at specific epoch
    class UnfreezeCallback(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            if trainer.current_epoch == unfreeze_epoch and freeze_backbone:
                print(f"\nUnfreezing backbone at epoch {unfreeze_epoch}")
                pl_module.model.unfreeze_backbone(
                    num_layers=2
                )  # Unfreeze last 2 layers

    callbacks.append(UnfreezeCallback())

    # Setup logger
    logger = TensorBoardLogger(
        "data/lightning_logs", name=f"resnet_encoder_{resnet_type}"
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=enable_progress_bar,
        log_every_n_steps=log_every_n_steps,
        accelerator="cpu" if torch.backends.mps.is_available() else "auto",
        devices=1 if torch.backends.mps.is_available() else "auto",
        deterministic=False,
        enable_checkpointing=True,
    )

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, data_module)

    return trainer, model, data_module


def main(dataset_to_load, resnet_type="resnet18"):
    """Main function to run the ResNet encoder training"""
    from neurodecoders.encoder.utils import (
        load_latest_data,
        plot_training_results,
        preprocess_data,
        save_predictions,
        visualize_data,
    )

    print(f"=== ResNet Neural Encoder Training ({resnet_type}) ===")

    # Load data
    print("Loading data...")
    images, firing_rates, data_file = load_latest_data(dataset_to_load)

    # Preprocess data
    images, firing_rates = preprocess_data(images, firing_rates)

    # Visualize data
    visualize_data(firing_rates)

    # Train with ResNet
    trainer, model, data_module = train_resnet_encoder(
        images=images,
        firing_rates=firing_rates,
        resnet_type=resnet_type,
        epochs=30,
        learning_rate=1e-3,
        freeze_backbone=True,
        unfreeze_epoch=15,
        enable_progress_bar=True,
    )

    # Plot training results
    plot_training_results(model.train_losses, model.val_losses)

    # Save predictions
    save_predictions(
        model.model, images, firing_rates, data_file, "data", dataset_to_load
    )

    # Save final model
    dataset_name = Path(data_file).stem
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = (
        f"data/resnet_encoder_model_{resnet_type}_{dataset_name}_datetime-"
        f"{timestamp}.pth"
    )
    torch.save(model.model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    print(f"=== ResNet Training Complete ({resnet_type}) ===")


if __name__ == "__main__":
    # Use the CIFAR-10 dataset
    dataset_to_load = Path(
        "data/synthdata_dataset-cifar10_sta-perlin_noise_patterns,11,11_n_neurons-1000_n_images-1000_datetime-20250703_162151.npz"
    )
    main(dataset_to_load, resnet_type="resnet18")
