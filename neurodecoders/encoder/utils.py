"""
Utility functions for neural encoder training and evaluation.

This module contains data loading, preprocessing, visualization, and
model saving/loading utilities used across different encoder architectures.
"""

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split


# ---- Dataset class ----
class NeuralDataset(Dataset):
    """Dataset for neural firing rate data paired with images."""

    def __init__(self, images, firing_rates, labels=None):
        # Recent synthetic datasets are consistently built with
        # (batch, channel, height, width) format
        self.images = torch.tensor(images, dtype=torch.float32)
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
        dataset_metadata=None,
        use_memory_mapping=False,
        chunk_size=10000,
        prefetch_factor=2,
        pin_memory=True,
    ):
        super().__init__()
        self.images = images
        self.firing_rates = firing_rates
        self.labels = labels
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Data loading configuration
        self.use_memory_mapping = use_memory_mapping
        self.chunk_size = chunk_size
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory

        # Dataset generation metadata
        self.dataset_metadata = dataset_metadata or {}

        # Core dataset attributes
        self.synthetic = True  # Always synthetic for this codebase
        self.timestamp = self.dataset_metadata.get("dataset_timestamp")
        self.git_commit = self.dataset_metadata.get("git_commit")
        self.git_branch = self.dataset_metadata.get("git_branch")

        # STA pattern information
        self.sta_pattern = self.dataset_metadata.get("sta_pattern")
        self.sta_patch_width = self.dataset_metadata.get("sta_patch_width")
        self.sta_patch_height = self.dataset_metadata.get("sta_patch_height")
        self.sta_type = self.dataset_metadata.get("sta_type")

        # Dataset configuration
        self.dataset_type = self.dataset_metadata.get("dataset_type")
        self.n_neurons = self.dataset_metadata.get("n_neurons")
        self.n_images = self.dataset_metadata.get("n_images")
        self.dataset_filename = self.dataset_metadata.get("dataset_filename")

        # Data statistics
        self.input_shape = images.shape if images is not None else None
        self.output_neurons = (
            firing_rates.shape[1] if firing_rates is not None else None
        )
        self.total_size_mb = self._calculate_total_size_mb()

        # Create full dataset
        self.full_dataset = NeuralDataset(images, firing_rates, labels)
        self.setup_splits()

    def _calculate_total_size_mb(self):
        """Calculate total dataset size in MB."""
        total_bytes = 0
        if self.images is not None:
            total_bytes += self.images.nbytes
        if self.firing_rates is not None:
            total_bytes += self.firing_rates.nbytes
        if self.labels is not None:
            total_bytes += self.labels.nbytes
        return total_bytes / (1024 * 1024)

    def get_metadata_summary(self):
        """Get a summary of dataset metadata as a dictionary."""
        return {
            "synthetic": self.synthetic,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "sta_pattern": self.sta_pattern,
            "sta_patch_width": self.sta_patch_width,
            "sta_patch_height": self.sta_patch_height,
            "sta_type": self.sta_type,
            "dataset_type": self.dataset_type,
            "n_neurons": self.n_neurons,
            "n_images": self.n_images,
            "dataset_filename": self.dataset_filename,
            "input_shape": str(self.input_shape) if self.input_shape else None,
            "output_neurons": self.output_neurons,
            "total_size_mb": self.total_size_mb,
        }

    def get_mlflow_parameters(self):
        """Get dataset metadata formatted for MLflow parameter logging."""
        metadata = self.get_metadata_summary()
        return {
            f"dataset_{k}": v for k, v in metadata.items() if v is not None
        }

    def get_dataset_id(self):
        """Generate a unique dataset identifier."""
        return (
            f"{self.dataset_type or 'unknown'}_"
            f"{self.sta_pattern or 'unknown'}_"
            f"{self.n_neurons or 'unknown'}n_"
            f"{self.n_images or 'unknown'}i"
        )

    def setup_splits(self):
        """Set up train/validation/test splits."""
        # Check if we're using pre-split data (from synthetic generation)
        if hasattr(self, "dataset_metadata") and self.dataset_metadata:
            data_split = self.dataset_metadata.get("data_split")
            if data_split == "train":
                # We're using train data, create validation split from it
                total_size = len(self.full_dataset)
                train_size = int(
                    total_size * 0.8
                )  # Use 80% of train data for training
                val_size = total_size - train_size  # Use 20% for validation

                self.train_dataset, self.val_dataset = random_split(
                    self.full_dataset,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(42),
                )
                # For test, we'll use the same as validation for now
                self.test_dataset = self.val_dataset
                print(
                    f"Using pre-split train data: "
                    f"{train_size} train, {val_size} val"
                )
                return

        # Default behavior for non-pre-split data
        total_size = len(self.full_dataset)
        train_size = int(total_size * self.train_split)
        val_size = int(total_size * self.val_split)
        test_size = total_size - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        """Return training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor
            if self.num_workers > 0
            else None,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        """Return validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor
            if self.num_workers > 0
            else None,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        """Return test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor
            if self.num_workers > 0
            else None,
            persistent_workers=self.num_workers > 0,
        )
