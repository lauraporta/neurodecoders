"""
Utility functions for neural encoder training and evaluation.

This module contains data loading, preprocessing, visualization, and
model saving/loading utilities used across different encoder architectures.
"""

import datetime
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split


# ---- Dataset class ----
class NeuralDataset(Dataset):
    """Dataset for neural firing rate data paired with images."""

    def __init__(self, images, firing_rates, labels=None):
        self.images = torch.tensor(
            images[:, None, :, :], dtype=torch.float32
        )  # Add channel dim if needed
        self.firing_rates = torch.tensor(firing_rates, dtype=torch.float32)
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.firing_rates[idx]


# ---- Data Module ----
class NeuralDataModule(pl.LightningDataModule):
    """Lightning data module for handling neural firing rate datasets."""

    def __init__(
        self,
        images,
        firing_rates,
        labels=None,
        train_split=0.7,
        val_split=0.15,
        batch_size=32,
        num_workers=0,
    ):
        super().__init__()
        self.images = images
        self.firing_rates = firing_rates
        self.labels = labels
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Create full dataset
        self.full_dataset = NeuralDataset(images, firing_rates, labels)
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


# ---- Data Loading Functions ----
def load_latest_data(dataset_to_load):
    """Load the latest neural data file"""
    # If dataset_to_load is a Path object, convert to string
    if hasattr(dataset_to_load, "__str__"):
        dataset_to_load = str(dataset_to_load)

    # Check if it's a specific file or a pattern
    if os.path.isfile(dataset_to_load):
        # It's a specific file
        files = [dataset_to_load]
    else:
        # It's a pattern, use glob
        files = glob.glob(dataset_to_load)

    if not files:
        raise FileNotFoundError(
            f"No neural data files found matching: {dataset_to_load}"
        )

    latest_file = max(files, key=os.path.getctime)
    data = np.load(latest_file)
    images = data["images"]  # Expecting shape: (N, 1, H, W)
    firing_rates = data[
        "responses"
    ]  # Shape: (N, C) - already in firing rate format

    return images, firing_rates, latest_file


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

    return images, firing_rates


# ---- Visualization Functions ----
def visualize_data(firing_rates):
    """Visualize the firing rate data"""
    print("\nFiring rate statistics:")
    print(f"Mean firing rate: {firing_rates.mean():.3f}")
    print(f"Std firing rate: {firing_rates.std():.3f}")
    print(f"Min firing rate: {firing_rates.min():.3f}")
    print(f"Max firing rate: {firing_rates.max():.3f}")

    # Plot firing rate distribution
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(firing_rates.flatten(), bins=50)
    plt.title("Firing Rate Distribution")
    plt.xlabel("Firing Rate")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.imshow(firing_rates[:100].T, aspect="auto", cmap="viridis")
    plt.colorbar(label="Firing Rate")
    plt.title("Firing Rates for First 100 Images")
    plt.xlabel("Image Index")
    plt.ylabel("Neuron Index")
    plt.tight_layout()
    plt.show()


def plot_training_results(train_losses, val_losses):
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
    plt.show()


# ---- Model Saving/Loading Functions ----
def save_predictions(
    model,
    images,
    firing_rates,
    input_file_path,
    output_dir="workspace/predictions/encoder",
    dataset_to_load=None,
):
    """Save predicted neural responses with the same timestamp as input file"""
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

    # Prepare images for prediction
    if images.ndim == 4:
        # Images are already [N, C, H, W]
        input_images = torch.tensor(images, dtype=torch.float32)
    else:
        # Images are [N, H, W], add channel dimension
        input_images = torch.tensor(images[:, None, :, :], dtype=torch.float32)

    # Run predictions in batches to avoid GPU memory issues
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    batch_size = 32  # Process images in smaller batches
    predictions_list = []
    total_batches = (len(input_images) + batch_size - 1) // batch_size

    print(
        f"Running inference on {len(input_images)} images in "
        f"{total_batches} batches..."
    )

    with torch.no_grad():
        for i, batch_start in enumerate(
            range(0, len(input_images), batch_size)
        ):
            batch = input_images[batch_start : batch_start + batch_size].to(
                device
            )
            batch_predictions = model(batch).cpu().numpy()
            predictions_list.append(batch_predictions)
            print(f"Processed batch {i + 1}/{total_batches}")

        # Clear GPU cache after inference
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    predictions = np.concatenate(predictions_list, axis=0)

    # Save predictions with same timestamp
    output_filename = f"encoder_predictions_{Path(input_file_path).stem}.npz"
    output_path = os.path.join(output_dir, output_filename)

    np.savez(
        output_path,
        predicted_responses=predictions,
        actual_responses=firing_rates,
        input_file=input_file_path,
        timestamp=timestamp,
    )

    print(f"Predictions saved to: {output_path}")
    print(f"Predicted responses shape: {predictions.shape}")
    print(f"Mean predicted firing rate: {predictions.mean():.3f}")

    return output_path
