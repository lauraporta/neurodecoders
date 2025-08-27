import argparse
import datetime
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader, Dataset, random_split

from neurodecoders.paths import get_path


# ---- Dataset class ----
class NeuralDecoderDataset(Dataset):
    def __init__(self, firing_rates, images, device=None):
        self.firing_rates = torch.tensor(firing_rates, dtype=torch.float32)
        self.images = torch.tensor(
            images[:, None, :, :], dtype=torch.float32
        )  # Add channel dim
        self.device = device

    def __len__(self):
        return len(self.firing_rates)

    def __getitem__(self, idx):
        firing_rate = self.firing_rates[idx]
        image = self.images[idx]

        # Move to device if specified
        if self.device is not None:
            firing_rate = firing_rate.to(self.device)
            image = image.to(self.device)

        return firing_rate, image


# ---- Model definition ----
class SimpleDecoder(nn.Module):
    def __init__(self, in_neurons, image_size=64):
        super().__init__()
        self.image_size = image_size

        # Fully connected layers to expand neural responses
        self.fc = nn.Sequential(
            nn.Linear(in_neurons, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 8 * 8 * 512),  # Reshape to 8x8x512
            nn.ReLU(),
        )

        # Upsampling + Convolution layers to fix checkerboard artifacts
        # This replaces ConvTranspose2d which can cause checkerboard patterns
        # due to uneven overlap in the upsampling process
        self.deconv = nn.Sequential(
            # 8x8 -> 16x16
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 16x16 -> 32x32
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 32x32 -> 64x64
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Final layer to get single channel
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),  # Output values between -1 and 1
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 8, 8)  # Reshape to 8x8x512
        x = self.deconv(x)
        # Use interpolation instead of adaptive pooling for MPS compatibility
        x = F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return x


# ---- Lightning Module ----
class DecoderLightningModule(pl.LightningModule):
    def __init__(
        self, in_neurons, image_size=64, learning_rate=1e-3, weight_decay=1e-5
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = SimpleDecoder(in_neurons, image_size)
        self.loss_fn = nn.MSELoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Store training history for plotting
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)

        # Log training loss
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)

        # Log validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)

        # Log test loss
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
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


# ---- Data Module ----
class DecoderDataModule(pl.LightningDataModule):
    def __init__(
        self,
        firing_rates,
        images,
        train_split=0.7,
        val_split=0.15,
        batch_size=32,
        num_workers=0,
        device=None,
    ):
        super().__init__()
        self.firing_rates = firing_rates
        self.images = images
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

        # Create full dataset
        self.full_dataset = NeuralDecoderDataset(firing_rates, images, device)
        self.setup_splits()

    def setup_splits(self):
        """Setup train/val/test splits"""
        total_size = len(self.full_dataset)
        train_size = int(self.train_split * total_size)
        val_size = int(self.val_split * total_size)
        test_size = total_size - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


def load_latest_data(dataset_to_load):
    """Load the latest neural data file"""
    # Convert Path object to string if needed
    if hasattr(dataset_to_load, "__str__"):
        dataset_to_load = str(dataset_to_load)

    # Handle specific filename - check if it's a full path or just filename
    if os.path.isabs(dataset_to_load):
        # Full path provided
        latest_file = dataset_to_load
    else:
        # Just filename provided - construct full path
        synthetic_dir = get_path("workspace/datasets/synthetic")
        latest_file = os.path.join(synthetic_dir, dataset_to_load)

    # Verify file exists
    if not os.path.exists(latest_file):
        raise FileNotFoundError(f"Data file not found: {latest_file}")

    data = np.load(latest_file)
    images = data["images"]  # Expecting shape: (N, 1, H, W) or (N, H, W)
    firing_rates = data["responses"]  # Shape: (N, C) - neural responses

    return images, firing_rates, latest_file


def print_device_info():
    """Print detailed information about available devices"""
    print("\n=== Device Information ===")

    if torch.cuda.is_available():
        print("CUDA available: True")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            device_capability = torch.cuda.get_device_capability(i)
            device_memory = (
                torch.cuda.get_device_properties(i).total_memory / 1024**3
            )  # Convert to GB

            print(f"\nDevice {i}: {device_name}")
            print(
                f"  Compute Capability: "
                f"{device_capability[0]}.{device_capability[1]}"
            )
            print(f"  Memory: {device_memory:.1f} GB")

            # Check for Tensor Cores
            if device_capability[0] >= 7:
                print("  Tensor Cores: Available (Volta+ architecture)")
            else:
                print("  Tensor Cores: Not available (pre-Volta architecture)")

    elif torch.backends.mps.is_available():
        print("MPS (Apple Silicon) available: True")
        print("Device: Apple Silicon GPU")
    else:
        print("CUDA available: False")
        print("MPS available: False")
        print("Using: CPU")

    print("=" * 30 + "\n")


def preprocess_data(images, firing_rates):
    """Preprocess and validate the data"""
    print(f"Images shape: {images.shape}")
    print(f"Firing rates shape: {firing_rates.shape}")

    # Handle 4D image input if present
    if images.ndim == 4:
        N, _, H, W = images.shape
        images = images[:, 0, :, :]  # Take first channel
    elif images.ndim == 3:
        N, H, W = images.shape
    else:
        raise ValueError(f"Unexpected image shape: {images.shape}")

    # Validate shapes
    N_r, C = firing_rates.shape
    if N != N_r:
        raise ValueError(
            f"Mismatch: images have {N} samples but firing rates have {N_r}"
        )

    return images, firing_rates, H, W


def visualize_data(firing_rates, images, save_path=None):
    """Visualize the neural response data and sample images"""
    print("\nFiring rate statistics:")
    print(f"Mean firing rate: {firing_rates.mean():.3f}")
    print(f"Std firing rate: {firing_rates.std():.3f}")
    print(f"Min firing rate: {firing_rates.min():.3f}")
    print(f"Max firing rate: {firing_rates.max():.3f}")

    # Plot firing rate distribution
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(firing_rates.flatten(), bins=50)
    plt.title("Firing Rate Distribution")
    plt.xlabel("Firing Rate")
    plt.ylabel("Count")

    plt.subplot(1, 3, 2)
    plt.imshow(firing_rates[:100].T, aspect="auto", cmap="viridis")
    plt.colorbar(label="Firing Rate")
    plt.title("Firing Rates for First 100 Images")
    plt.xlabel("Image Index")
    plt.ylabel("Neuron Index")

    plt.subplot(1, 3, 3)
    plt.imshow(images[0], cmap="gray")
    plt.title("Sample Image")
    plt.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Data visualization saved to: {save_path}")

    plt.show()


def train_model_lightning(
    firing_rates,
    images,
    train_split=0.7,
    val_split=0.15,
    batch_size=32,
    learning_rate=1e-3,
    epochs=30,
    enable_progress_bar=True,
    log_every_n_steps=50,
    callbacks=None,
):
    """
    Train decoder using PyTorch Lightning

    Returns:
        trainer: The trained trainer object
        model: The trained model
        data_module: The data module
    """
    # Check available devices
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)

        # Check if device supports Tensor Cores and enable them
        if (
            torch.cuda.get_device_capability(0)[0] >= 7
        ):  # Volta architecture and newer
            print(
                f"Tensor Cores detected on {device_name}. Enabling high "
                f"precision matmul for optimal performance."
            )
            torch.set_float32_matmul_precision("high")
        else:
            print(
                f"CUDA device {device_name} detected, but Tensor Cores not "
                f"available."
            )

    elif torch.backends.mps.is_available():
        device = "MPS (Apple Silicon)"
        device_name = "Apple Silicon GPU"
    else:
        device = "CPU"
        device_name = "CPU"

    print(f"Training on: {device} - {device_name}")

    # Create data module
    data_module = DecoderDataModule(
        firing_rates=firing_rates,
        images=images,
        train_split=train_split,
        val_split=val_split,
        batch_size=batch_size,
        device=device
        if device == "CUDA"
        else None,  # Only pass device for CUDA, let Lightning handle others
    )

    # Create model
    model = DecoderLightningModule(
        in_neurons=firing_rates.shape[1],
        image_size=images.shape[1],  # Assuming square images
        learning_rate=learning_rate,
    )

    # Setup callbacks
    if callbacks is None:
        callbacks = []

    # Add default callbacks
    callbacks.extend(
        [
            LearningRateMonitor(logging_interval="epoch"),
        ]
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        enable_progress_bar=enable_progress_bar,
        log_every_n_steps=log_every_n_steps,
        accelerator="auto",  # Let Lightning automatically detect the best
        # accelerator
        devices="auto",  # Let Lightning automatically detect the number of
        # devices
        deterministic=False,
        enable_checkpointing=False,  # Disable checkpoints
    )

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, data_module)

    return trainer, model, data_module


def plot_training_results(train_losses, val_losses, save_path=None):
    """Plot training results"""
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to: {save_path}")

    plt.show()


def save_predictions(
    model,
    firing_rates,
    images,
    input_file_path,
    output_dir=get_path("workspace/predictions/decoder"),
    dataset_to_load=None,
):
    """Save model predictions and reconstructed images"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract timestamp from input filename
    # Expected format:
    # simulated_neural_data_*neurons_*images_YYYYMMDD_HHMMSS.npz
    timestamp_match = re.search(r"(\d{8}_\d{6})\.npz$", input_file_path)
    if timestamp_match:
        timestamp = timestamp_match.group(1)
    else:
        # If no timestamp found, use current time
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate predictions
    model.eval()
    with torch.no_grad():
        firing_rates_tensor = torch.tensor(firing_rates, dtype=torch.float32)
        predictions = model(firing_rates_tensor)
        predictions = predictions.squeeze().cpu().numpy()

    # Ensure images are in the correct format for visualization
    if images.ndim == 4:
        # Images are (N, C, H, W) - remove channel dimension
        images_vis = images.squeeze(1)  # Remove channel dimension
    elif images.ndim == 3:
        # Images are already (N, H, W)
        images_vis = images
    else:
        raise ValueError(f"Unexpected image shape: {images.shape}")

    # Ensure predictions are in the correct format
    if predictions.ndim == 4:
        # Predictions are (N, C, H, W) - remove channel dimension
        predictions_vis = predictions.squeeze(1)
    elif predictions.ndim == 3:
        # Predictions are already (N, H, W)
        predictions_vis = predictions
    else:
        raise ValueError(f"Unexpected prediction shape: {predictions.shape}")

    # Handle dataset_to_load parameter
    if dataset_to_load is not None:
        # Convert to Path object if it's a string
        if isinstance(dataset_to_load, str):
            dataset_to_load = Path(dataset_to_load)
        dataset_name = dataset_to_load.stem
    else:
        # Extract name from input_file_path if dataset_to_load is None
        input_path = Path(input_file_path)
        dataset_name = input_path.stem

    # Save predictions with same timestamp
    output_filename = f"decoder_predictions_{dataset_name}.npz"
    output_path = os.path.join(output_dir, output_filename)

    np.savez(
        output_path,
        original_images=images_vis,
        reconstructed_images=predictions_vis,
        neural_responses=firing_rates,
        input_file=input_file_path,
        timestamp=timestamp,
    )

    print(f"Predictions saved to: {output_path}")
    print(f"Reconstructed images shape: {predictions_vis.shape}")
    print(
        f"Mean reconstruction error: \
            {np.mean((predictions_vis - images_vis) ** 2):.4f}"
    )

    # Visualize some reconstructions
    n_samples = min(10, len(images_vis))
    fig, axes = plt.subplots(2, n_samples, figsize=(2 * n_samples, 4))

    for i in range(n_samples):
        # Original image
        axes[0, i].imshow(images_vis[i], cmap="gray")
        axes[0, i].set_title(f"Original {i + 1}")
        axes[0, i].axis("off")

        # Reconstructed image
        axes[1, i].imshow(predictions_vis[i], cmap="gray")
        axes[1, i].set_title(f"Reconstructed {i + 1}")
        axes[1, i].axis("off")

    plt.tight_layout()

    # Save reconstruction plot to plots directory
    plots_dir = get_path("workspace/plots/decoder")
    os.makedirs(plots_dir, exist_ok=True)
    reconstruction_plot_path = os.path.join(
        plots_dir, f"decoder_reconstructions_{timestamp}.png"
    )
    plt.savefig(reconstruction_plot_path, dpi=150, bbox_inches="tight")
    print(f"Reconstruction plot saved to: {reconstruction_plot_path}")

    plt.show()

    return output_path


def save_test_set_decoded_images(
    model,
    data_module,
    output_dir=get_path("workspace/predictions/decoder"),
    num_samples=5,
    dataset_to_load=None,
):
    """Save decoded images from the test set"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get test set data
    test_dataloader = data_module.test_dataloader()

    # Get a few samples from test set
    model.eval()
    test_images = []
    test_firing_rates = []
    test_predictions = []

    with torch.no_grad():
        for i, (firing_rates, images) in enumerate(test_dataloader):
            if i >= num_samples:
                break

            # Generate predictions
            predictions = model(firing_rates)

            # Store data
            test_firing_rates.append(firing_rates.cpu().numpy())
            test_images.append(images.cpu().numpy())
            test_predictions.append(predictions.cpu().numpy())

    # Concatenate all samples
    test_firing_rates = np.concatenate(test_firing_rates, axis=0)
    test_images = np.concatenate(test_images, axis=0)
    test_predictions = np.concatenate(test_predictions, axis=0)

    # Ensure proper shapes for visualization
    if test_images.ndim == 4:
        test_images_vis = test_images.squeeze(1)  # Remove channel dimension
    else:
        test_images_vis = test_images

    if test_predictions.ndim == 4:
        test_predictions_vis = test_predictions.squeeze(
            1
        )  # Remove channel dimension
    else:
        test_predictions_vis = test_predictions

    # Handle dataset_to_load parameter
    if dataset_to_load is not None:
        if isinstance(dataset_to_load, str):
            dataset_to_load = Path(dataset_to_load)
        dataset_name = dataset_to_load.stem
    else:
        dataset_name = "test_set"

    # Save test set predictions
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"test_set_decoded_{dataset_name}_{timestamp}.npz"
    output_path = os.path.join(output_dir, output_filename)

    np.savez(
        output_path,
        original_images=test_images_vis,
        decoded_images=test_predictions_vis,
        neural_responses=test_firing_rates,
        dataset_name=dataset_name,
        timestamp=timestamp,
    )

    print(f"Test set decoded images saved to: {output_path}")
    print(f"Number of test samples: {len(test_images_vis)}")
    print(
        f"Mean reconstruction error: "
        f"{np.mean((test_predictions_vis - test_images_vis) ** 2):.4f}"
    )

    # Visualize test set reconstructions
    n_samples = min(num_samples, len(test_images_vis))
    fig, axes = plt.subplots(3, n_samples, figsize=(2 * n_samples, 6))

    for i in range(n_samples):
        # Original image
        axes[0, i].imshow(test_images_vis[i], cmap="gray")
        axes[0, i].set_title(f"Original {i + 1}")
        axes[0, i].axis("off")

        # Decoded image
        axes[1, i].imshow(test_predictions_vis[i], cmap="gray")
        axes[1, i].set_title(f"Decoded {i + 1}")
        axes[1, i].axis("off")

        # Difference image
        diff = np.abs(test_predictions_vis[i] - test_images_vis[i])
        axes[2, i].imshow(diff, cmap="hot")
        axes[2, i].set_title(f"Difference {i + 1}")
        axes[2, i].axis("off")

    plt.tight_layout()

    # Save test set reconstruction plot
    plots_dir = get_path("workspace/plots/decoder")
    os.makedirs(plots_dir, exist_ok=True)
    test_plot_path = os.path.join(
        plots_dir, f"test_set_decoded_{dataset_name}_{timestamp}.png"
    )
    plt.savefig(test_plot_path, dpi=150, bbox_inches="tight")
    print(f"Test set reconstruction plot saved to: {test_plot_path}")

    plt.show()

    return output_path


def main(dataset_to_load, epochs=100):
    """Main function to run the decoder training"""
    print("=== Neural Decoder Training with PyTorch Lightning ===")
    print(f"Training for {epochs} epochs")

    # Print device information
    print_device_info()

    # Load data
    print("Loading data...")
    images, firing_rates, data_file = load_latest_data(dataset_to_load)

    # Preprocess data
    images, firing_rates, H, W = preprocess_data(images, firing_rates)

    # Create plots directory
    plots_dir = get_path("workspace/plots/decoder")
    os.makedirs(plots_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Visualize data
    data_viz_path = os.path.join(
        plots_dir, f"data_visualization_{timestamp}.png"
    )
    visualize_data(firing_rates, images, save_path=data_viz_path)

    # Train with Lightning
    _, model, data_module = train_model_lightning(
        firing_rates=firing_rates,
        images=images,
        epochs=epochs,
        learning_rate=1e-4,
    )

    # Plot training results
    training_curves_path = os.path.join(
        plots_dir, f"training_curves_{timestamp}.png"
    )
    plot_training_results(
        model.train_losses, model.val_losses, save_path=training_curves_path
    )

    # Save predictions
    save_predictions(
        model,
        firing_rates,
        images,
        data_file,
        output_dir=get_path("workspace/predictions/decoder"),
        dataset_to_load=dataset_to_load,
    )

    # Save test set decoded images
    save_test_set_decoded_images(
        model,
        data_module,
        output_dir=get_path("workspace/predictions/decoder"),
        num_samples=5,
        dataset_to_load=dataset_to_load,
    )

    # Save final model
    if isinstance(dataset_to_load, str):
        dataset_to_load = Path(dataset_to_load)
    model_dir = get_path("workspace/models/decoders")
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/decoder_{dataset_to_load.stem}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    print("=== Lightning Training Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train neural decoder using PyTorch Lightning"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Filename of the dataset to load",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )

    args = parser.parse_args()

    # Use the provided dataset filename directly
    dataset_to_load = args.dataset

    main(dataset_to_load, args.epochs)
