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


def plot_training_curves(train_losses, val_losses):
    """Plot training and validation loss curves"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label="Train Loss", linewidth=2)
    ax.plot(val_losses, label="Validation Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_firing_rate_distribution(firing_rates):
    """Plot firing rate distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Histogram
    ax1.hist(
        firing_rates.flatten(),
        bins=50,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    ax1.set_title("Firing Rate Distribution")
    ax1.set_xlabel("Firing Rate (Hz)")
    ax1.set_ylabel("Count")
    ax1.grid(True, alpha=0.3)

    # Heatmap
    im = ax2.imshow(firing_rates[:100].T, aspect="auto", cmap="viridis")
    ax2.set_title("Firing Rates for First 100 Images")
    ax2.set_xlabel("Image Index")
    ax2.set_ylabel("Neuron Index")
    plt.colorbar(im, ax=ax2, label="Firing Rate (Hz)")

    plt.tight_layout()
    return fig


def plot_predictions_vs_actual(pred, actual, n_samples=10):
    """Plot predicted vs actual firing rates"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    # Get the number of neurons (columns) in the data
    n_neurons = min(pred.shape[1], actual.shape[1])
    n_plots = min(n_samples, len(axes), n_neurons)

    for i in range(n_plots):
        # Plot all samples for this neuron
        axes[i].scatter(actual[:, i], pred[:, i], alpha=0.6, s=20)

        # Add diagonal line
        max_val = max(actual[:, i].max(), pred[:, i].max())
        axes[i].plot([0, max_val], [0, max_val], "r--", alpha=0.8)

        axes[i].set_xlabel("Actual Firing Rate")
        axes[i].set_ylabel("Predicted Firing Rate")
        axes[i].set_title(f"Neuron {i + 1}")
        axes[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig


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


def save_model_with_metadata(
    model,
    data_file,
    model_type="encoder",
    model_dir="workspace/models",
    additional_info=None,
):
    """
    Save model with descriptive filename including metadata.

    Args:
        model: The trained model to save
        data_file: Path to the dataset file used for training
        model_type: Type of model (e.g., "simple_encoder", "resnet18")
        model_dir: Directory to save the model
        additional_info: Dict with additional info to include in filename
    """
    os.makedirs(model_dir, exist_ok=True)

    # Extract dataset name
    dataset_name = Path(data_file).stem

    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build filename with metadata
    filename_parts = [model_type, dataset_name, f"datetime-{timestamp}"]

    if additional_info:
        for key, value in additional_info.items():
            filename_parts.append(f"{key}-{value}")

    model_filename = "_".join(filename_parts) + ".pth"
    model_path = os.path.join(model_dir, model_filename)

    # Save model state dict
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to: {model_path}")
    return model_path


def load_model_from_path(model_class, model_path, model_kwargs=None):
    """
    Load a model from a saved state dict.

    Args:
        model_class: The model class to instantiate
        model_path: Path to the saved model state dict
        model_kwargs: Keyword arguments for model instantiation

    Returns:
        Loaded model
    """
    if model_kwargs is None:
        model_kwargs = {}

    # Create model instance
    model = model_class(**model_kwargs)

    # Load state dict
    state_dict = torch.load(model_path, map_location="cpu")

    # Handle state dicts that have "model." prefix (from Lightning modules)
    if any(key.startswith("model.") for key in state_dict.keys()):
        # Strip the "model." prefix from all keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_key = key[6:]  # Remove "model." prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.eval()

    return model
