#!/usr/bin/env python3
"""
Generic MLflow training script for neural encoders.

This script provides a flexible interface for training different encoder models
with various configurations, datasets, and hyperparameters while tracking
experiments with MLflow.
"""

import argparse
import datetime
import os
import sys
from typing import Any, Dict, Tuple

# Add the encoder directory to the path
sys.path.append(os.path.dirname(__file__))

import numpy as np
import torch

from neurodecoders.encoder.models import (
    ResNetEncoder,
    SimpleEncoder,
    SimpleEncoderWithSkipConnection,
)
from neurodecoders.encoder.training import train_encoder
from neurodecoders.encoder.utils import NeuralDataModule
from neurodecoders.paths import get_path


# Configuration validation functions
def validate_config(config):
    """Simple config validation - removed complex logic"""
    pass


def parse_dataset_metadata(filename: str) -> Dict[str, Any]:
    """
    Parse dataset metadata from synthetic data filename.

    Expected format: synthdata_dataset-{dataset_type}_sta-{sta_type}_n_neurons-
    {n_neurons}_n_images-{n_images}.npz

    Args:
        filename: Synthetic data filename

    Returns:
        Dictionary containing parsed metadata
    """
    metadata = {}

    try:
        # Remove .npz extension
        name = filename.replace(".npz", "")

        # Parse dataset type
        if "dataset-" in name:
            dataset_part = name.split("dataset-")[1].split("_")[0]
            metadata["dataset_type"] = dataset_part

        # Parse STA type and parameters
        if "sta-" in name:
            sta_part = name.split("sta-")[1].split("_n_neurons")[0]
            metadata["sta_type"] = sta_part

            # Parse STA parameters if present
            if "," in sta_part:
                sta_parts = sta_part.split(",")
                metadata["sta_pattern"] = sta_parts[0]
                if len(sta_parts) >= 3:
                    metadata["sta_patch_width"] = str(int(sta_parts[1]))
                    metadata["sta_patch_height"] = str(int(sta_parts[2]))

        # Parse number of neurons
        if "n_neurons-" in name:
            neurons_part = name.split("n_neurons-")[1].split("_")[0]
            metadata["n_neurons"] = str(int(neurons_part))

        # Parse number of images
        if "n_images-" in name:
            images_part = name.split("n_images-")[1].split("_")[0]
            metadata["n_images"] = str(int(images_part))

    except Exception as e:
        print(
            f"Warning: Could not parse metadata from filename {filename}: {e}"
        )

    return metadata


def load_synthetic_data_from_workspace(
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load synthetic data from workspace/datasets/synthetic based on
    configuration.

    Args:
        config: Configuration dictionary with data parameters

    Returns:
        images, firing_rates, labels, metadata: Synthetic data and metadata
    """
    synthetic_dir = get_path("workspace/datasets/synthetic")

    if not os.path.exists(synthetic_dir):
        raise FileNotFoundError(
            f"Synthetic data directory {synthetic_dir} not found. "
            "Please run the synthetic data generation first."
        )

    # Get available synthetic data files
    available_files = [
        f for f in os.listdir(synthetic_dir) if f.endswith(".npz")
    ]

    if not available_files:
        raise FileNotFoundError(
            f"No synthetic data files found in {synthetic_dir}. "
            "Please run the synthetic data generation first."
        )

    # Required exact matches from config
    n_neurons = str(int(config["n_neurons"]))
    n_images = str(int(config["n_images"]))

    # Filter files that exactly match both counts
    matching = []
    for fname in available_files:
        meta = parse_dataset_metadata(fname)
        if (
            meta.get("n_neurons") == n_neurons
            and meta.get("n_images") == n_images
        ):
            fpath = os.path.join(synthetic_dir, fname)
            try:
                mtime = os.path.getmtime(fpath)
            except OSError:
                mtime = 0.0
            matching.append((mtime, fname))

    if not matching:
        raise ValueError(
            "No synthetic dataset matches the requested counts. "
            f"Requested n_neurons={n_neurons}, n_images={n_images}."
        )

    # Select the latest by modification time
    matching.sort(key=lambda x: x[0], reverse=True)
    selected_file = matching[0][1]

    # Parse metadata from filename
    metadata = parse_dataset_metadata(selected_file)

    # Add timestamp for dataset identification
    metadata["dataset_timestamp"] = datetime.datetime.now().isoformat()
    metadata["dataset_filename"] = selected_file

    # Load the data
    file_path = os.path.join(synthetic_dir, selected_file)
    print(f"Loading synthetic data from: {file_path}")

    try:
        # Check if memory mapping should be used
        use_memory_mapping = config["use_memory_mapping"]

        if use_memory_mapping:
            print(f"Loading data with memory mapping: {file_path}")
            data = np.load(file_path, mmap_mode="r")
        else:
            data = np.load(file_path)

        # Extract images, responses (firing rates), and labels
        if "images" in data and "responses" in data:
            images = data["images"]
            firing_rates = data["responses"]

            # Extract labels if available
            labels = data.get("labels", None)

            if labels is not None:
                print(
                    f"Loaded data: {images.shape} images, "
                    f"{firing_rates.shape} firing rates, "
                    f"{len(labels)} labels"
                )
            else:
                print(
                    f"Loaded data: {images.shape} images, "
                    f"{firing_rates.shape} firing rates "
                    "(no labels available)"
                )
        else:
            raise ValueError(
                "Invalid synthetic data format: missing 'images' or "
                "'responses'"
            )

        # Calculate memory usage
        total_memory_mb = (
            images.nbytes
            + firing_rates.nbytes
            + (labels.nbytes if labels is not None else 0)
        ) / (1024 * 1024)

        print(f"Dataset memory usage: {total_memory_mb:.1f} MB")

        if total_memory_mb > 1000 and not use_memory_mapping:
            print(
                "Warning: Large dataset detected. "
                "Consider using --use-memory-mapping"
            )

        return images, firing_rates, labels, metadata

    except Exception as e:
        raise RuntimeError(
            f"Error loading synthetic data from {file_path}: {e}"
        )


def get_model(config: Dict[str, Any]) -> torch.nn.Module:
    """
    Create model based on configuration.

    Args:
        config: Configuration dictionary with model parameters

    Returns:
        model: PyTorch model
    """
    model_type = config["model_type"]
    out_neurons = config.get("out_neurons")

    if out_neurons is None:
        raise ValueError(
            "out_neurons must be specified or inferred from dataset before "
            "calling get_model"
        )

    if model_type == "simple":
        return SimpleEncoder(out_neurons=out_neurons)
    elif model_type == "skip":
        return SimpleEncoderWithSkipConnection(out_neurons=out_neurons)
    elif model_type == "resnet":
        resnet_type = config["resnet_type"]
        freeze_backbone = config["freeze_backbone"]
        return ResNetEncoder(
            out_neurons=out_neurons,
            resnet_type=resnet_type,
            freeze_backbone=freeze_backbone,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train_with_config(config: Dict[str, Any]):
    """
    Train encoder with given configuration.

    Args:
        config: Configuration dictionary

    Returns:
        trainer, model, data_module: Training results
    """
    print("=== ENCODER TRAINING WITH CONFIG ===")
    print(f"Model type: {config['model_type']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Optimizer: {config['optimizer']}")
    print(f"Loss function: {config['loss_function']}")
    print(f"Scheduler: {config['scheduler']}")
    print(f"Dataset: {config['dataset_type']}")
    print(f"STA type: {config['sta_type']}")
    print(f"Neurons: {config['n_neurons']}")
    print(f"Images: {config['n_images']}")
    print(f"Mixed precision: {config['enable_mixed_precision']}")
    print(f"Early stopping: {config['enable_early_stopping']}")
    print(f"Checkpointing: {config['enable_checkpointing']}")

    # Load data
    images, firing_rates, labels, metadata = (
        load_synthetic_data_from_workspace(config)
    )

    # Infer out_neurons from dataset if not specified
    if config.get("out_neurons") is None:
        config["out_neurons"] = firing_rates.shape[1]
        print(f"Inferred output neurons from dataset: {config['out_neurons']}")
    else:
        print(f"Output neurons: {config['out_neurons']}")

    # Create data module
    data_module = NeuralDataModule(
        images=images,
        firing_rates=firing_rates,
        labels=labels,
        batch_size=config["batch_size"],
        dataset_metadata=metadata,
        use_memory_mapping=config["use_memory_mapping"],
        chunk_size=config["chunk_size"],
        prefetch_factor=config["prefetch_factor"],
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )

    # Create optimizer and scheduler configurations
    optimizer_config = {
        "type": config["optimizer"],
    }

    scheduler_config = {
        "type": config["scheduler"],
        "step_size": config["scheduler_step_size"],
        "gamma": config["scheduler_gamma"],
    }

    # Create model
    model = get_model(config)

    # Train the model using the main training function
    trainer, lightning_model, _ = train_encoder(
        model=model,
        data_module=data_module,
        model_name=f"{config['model_type']}_encoder",
        learning_rate=config["learning_rate"],
        epochs=config["epochs"],
        optimizer_config=optimizer_config,
        loss_fn=config["loss_function"],
        scheduler_config=scheduler_config,
        enable_mlflow=config["enable_mlflow"],
        mlflow_experiment_name=config["mlflow_experiment_name"],
        mlflow_run_name=config["mlflow_run_name"],
        # Enhanced training options
        enable_mixed_precision=config["enable_mixed_precision"],
        enable_early_stopping=config["enable_early_stopping"],
        early_stopping_patience=config["early_stopping_patience"],
        enable_checkpointing=config["enable_checkpointing"],
        gradient_clip_val=config["gradient_clip_val"],
    )

    return trainer, lightning_model, data_module


def main():
    """Main function for command-line training."""
    parser = argparse.ArgumentParser(
        description="Train neural encoder with MLflow tracking"
    )

    # Model configuration
    parser.add_argument(
        "--model-type",
        choices=["simple", "skip", "resnet"],
        default="simple",
        help="Type of encoder model",
    )
    parser.add_argument(
        "--resnet-type",
        choices=["resnet18", "resnet34", "resnet50"],
        default="resnet18",
        help="ResNet type (only for resnet model)",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        default=True,
        help="Freeze ResNet backbone (only for resnet model)",
    )
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="Unfreeze ResNet backbone (overrides --freeze-backbone)",
    )

    # Training configuration
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,  # Updated default
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "sgd", "adamw"],
        default="adam",
        help="Optimizer type",
    )
    parser.add_argument(
        "--loss-function",
        choices=["mse", "l1", "smooth_l1", "huber"],
        default="mse",
        help="Loss function",
    )
    parser.add_argument(
        "--scheduler",
        choices=["none", "step", "cosine", "plateau"],
        default="none",
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--scheduler-step-size",
        type=int,
        default=30,
        help="Step size for step scheduler",
    )
    parser.add_argument(
        "--scheduler-gamma",
        type=float,
        default=0.1,
        help="Gamma for step scheduler (multiplies LR by this factor)",
    )

    # Data configuration
    parser.add_argument(
        "--dataset-type",
        default="cifar10",
        help="Dataset type (cifar10, mnist, etc.)",
    )
    parser.add_argument(
        "--sta-type",
        default="periodic_patterns,70,70",
        help="STA pattern type",
    )
    parser.add_argument(
        "--n-neurons",
        type=int,
        default=100,
        help="Number of neurons in synthetic data",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=10000,
        help="Number of images in synthetic data",
    )

    # Data loading configuration
    parser.add_argument(
        "--use-memory-mapping",
        action="store_true",
        help="Use memory mapping for large datasets",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Chunk size for data loading",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Number of batches to prefetch in background (0=disable)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of subprocesses for data loading (0=main process)",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        default=True,
        help="Pin memory for faster GPU transfer",
    )
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable pin memory (overrides --pin-memory)",
    )

    # MLflow configuration
    parser.add_argument(
        "--experiment-name",
        default="neural_encoder",
        help="MLflow experiment name",
    )
    parser.add_argument("--run-name", help="MLflow run name")

    # Enhanced training options
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        default=True,
        help="Enable mixed precision training (16-bit)",
    )
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision training",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        default=True,
        help="Enable early stopping",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=100,  # Updated default
        help="Patience for early stopping",
    )
    parser.add_argument(
        "--checkpointing",
        action="store_true",
        default=True,
        help="Enable model checkpointing",
    )
    parser.add_argument(
        "--no-checkpointing",
        action="store_true",
        help="Disable model checkpointing",
    )
    parser.add_argument(
        "--gradient-clip-val",
        type=float,
        default=1.0,
        help="Gradient clipping value",
    )

    args = parser.parse_args()

    # Handle freeze_backbone logic
    freeze_backbone = args.freeze_backbone and not args.unfreeze_backbone

    # Handle pin_memory logic
    pin_memory = args.pin_memory and not args.no_pin_memory

    # Handle enhanced training options
    enable_mixed_precision = (
        args.mixed_precision and not args.no_mixed_precision
    )
    enable_early_stopping = args.early_stopping and not args.no_early_stopping
    enable_checkpointing = args.checkpointing and not args.no_checkpointing

    # Create config from command line arguments
    config = {
        "model_type": args.model_type,
        "out_neurons": None,  # Will be inferred from dataset
        "resnet_type": args.resnet_type,
        "freeze_backbone": freeze_backbone,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "loss_function": args.loss_function,
        "scheduler": args.scheduler,
        "scheduler_step_size": args.scheduler_step_size,
        "scheduler_gamma": args.scheduler_gamma,
        "dataset_type": "cifar10",  # Default dataset
        "sta_type": args.sta_type,
        "n_neurons": args.n_neurons,
        "n_images": args.n_images,
        "use_memory_mapping": args.use_memory_mapping,
        "chunk_size": args.chunk_size,
        "prefetch_factor": args.prefetch_factor,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "mlflow_experiment_name": args.experiment_name,
        "mlflow_run_name": args.run_name,
        # Enhanced training options
        "enable_mixed_precision": enable_mixed_precision,
        "enable_early_stopping": enable_early_stopping,
        "early_stopping_patience": args.early_stopping_patience,
        "enable_checkpointing": enable_checkpointing,
        "gradient_clip_val": args.gradient_clip_val,
        # MLflow toggle
        "enable_mlflow": True,
    }

    # Validate config
    validate_config(config)

    trainer, model, _ = train_with_config(config)

    print("\nTraining completed!")
    if model.train_losses:
        print(f"Final train loss: {model.train_losses[-1]:.4f}")
    if model.val_losses:
        print(f"Final validation loss: {model.val_losses[-1]:.4f}")

    print("\nTo view MLflow experiments, run:")
    print("mlflow ui")
    print("Then open http://localhost:5000 in your browser.")


if __name__ == "__main__":
    main()
