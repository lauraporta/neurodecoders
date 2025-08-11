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
from neurodecoders.encoder.training import (
    train_resnet_encoder,
    train_simple_encoder,
    train_skip_connection_encoder,
)
from neurodecoders.encoder.utils import NeuralDataModule, preprocess_data


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
    synthetic_dir = "workspace/datasets/synthetic"

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

    # Parse configuration to find matching file
    dataset_type = config.get("dataset_type", "cifar10")
    sta_type = config.get("sta_type", "perlin_noise_patterns,11,11")
    n_neurons = config.get("n_neurons", 1000)
    n_images = config.get("n_images", 1000)

    # Look for exact match first
    target_filename = (
        f"synthdata_dataset-{dataset_type}_sta-{sta_type}_n_neurons-"
        f"{n_neurons}_n_images-{n_images}"
    )

    matching_files = [f for f in available_files if target_filename in f]

    if not matching_files:
        # If no exact match, find the closest match
        print(f"Warning: No exact match found for {target_filename}")
        print("Available files:")
        for f in available_files:
            print(f"  {f}")

        # Try to find any file with the same dataset_type and sta_type
        partial_matches = [
            f
            for f in available_files
            if f"dataset-{dataset_type}" in f and f"sta-{sta_type}" in f
        ]

        if partial_matches:
            # Use the most recent file
            partial_matches.sort(reverse=True)
            selected_file = partial_matches[0]
            print(f"Using closest match: {selected_file}")
        else:
            # Use the most recent file overall
            available_files.sort(reverse=True)
            selected_file = available_files[0]
            print(f"Using most recent file: {selected_file}")
    else:
        # Use the most recent exact match
        matching_files.sort(reverse=True)
        selected_file = matching_files[0]
        print(f"Using exact match: {selected_file}")

    # Parse metadata from filename
    metadata = parse_dataset_metadata(selected_file)

    # Add timestamp for dataset identification
    metadata["dataset_timestamp"] = datetime.datetime.now().isoformat()
    metadata["dataset_filename"] = selected_file

    # Load the data
    file_path = os.path.join(synthetic_dir, selected_file)
    print(f"Loading synthetic data from: {file_path}")

    try:
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
    model_type = config.get("model_type", "simple")
    out_neurons = config.get("out_neurons", 100)

    if model_type == "simple":
        return SimpleEncoder(out_neurons=out_neurons)
    elif model_type == "skip":
        return SimpleEncoderWithSkipConnection(out_neurons=out_neurons)
    elif model_type == "resnet":
        resnet_type = config.get("resnet_type", "resnet18")
        freeze_backbone = config.get("freeze_backbone", True)
        return ResNetEncoder(
            out_neurons=out_neurons,
            resnet_type=resnet_type,
            freeze_backbone=freeze_backbone,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract training configuration from config.

    Args:
        config: Full configuration dictionary

    Returns:
        training_config: Training-specific configuration
    """
    return {
        "learning_rate": config.get("learning_rate", 1e-3),
        "weight_decay": config.get("weight_decay", 1e-5),
        "epochs": config.get("epochs", 30),
        "batch_size": config.get("batch_size", 32),
        "train_split": config.get("train_split", 0.7),
        "val_split": config.get("val_split", 0.15),
        "unfreeze_epoch": config.get("unfreeze_epoch"),
        "optimizer_config": config.get("optimizer_config"),
    }


def get_mlflow_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract MLflow configuration from config.

    Args:
        config: Full configuration dictionary

    Returns:
        mlflow_config: MLflow-specific configuration
    """
    return {
        "enable_mlflow": config.get("enable_mlflow", True),
        "mlflow_experiment_name": config.get(
            "mlflow_experiment_name", "neural_encoder"
        ),
        "mlflow_run_name": config.get("mlflow_run_name"),
        "mlflow_tracking_uri": config.get("mlflow_tracking_uri"),
    }


def train_with_config(config: Dict[str, Any]) -> tuple:
    """
    Train model with given configuration.

    Args:
        config: Complete configuration dictionary

    Returns:
        trainer, model, data_module: Training results
    """
    print("Training with configuration:")
    print(f"  Model: {config.get('model_type', 'simple')}")
    print(
        f"  Dataset: {config.get('dataset_type', 'cifar10')} + "
        f"{config.get('sta_type', 'perlin_noise_patterns,11,11')}"
    )
    print(f"  Neurons: {config.get('out_neurons', 100)}")
    print(f"  Epochs: {config.get('epochs', 30)}")
    print(f"  Learning Rate: {config.get('learning_rate', 1e-3)}")

    # Always load synthetic data from workspace
    images, firing_rates, labels, dataset_metadata = (
        load_synthetic_data_from_workspace(config)
    )

    # Ensure data matches model configuration
    out_neurons = config.get("out_neurons", 100)
    if firing_rates.shape[1] != out_neurons:
        if firing_rates.shape[1] > out_neurons:
            # Truncate to match model - warn about waste
            print(
                f"WARNING: Truncating firing rates from "
                f"{firing_rates.shape[1]} to {out_neurons} neurons. "
                f"This wastes {firing_rates.shape[1] - out_neurons} neurons."
            )
            firing_rates = firing_rates[:, :out_neurons]
        else:
            # Raise error for insufficient neurons
            raise ValueError(
                f"Model expects {out_neurons} neurons but dataset only has "
                f"{firing_rates.shape[1]} neurons. Please either: "
                f"1) Reduce model out_neurons to {firing_rates.shape[1]}, or "
                f"2) Generate synthetic data with more neurons."
            )

    # Preprocess data
    images, firing_rates = preprocess_data(images, firing_rates)

    # Create data module
    training_config = get_training_config(config)
    data_module = NeuralDataModule(
        images=images,
        firing_rates=firing_rates,
        labels=labels,
        train_split=training_config["train_split"],
        val_split=training_config["val_split"],
        batch_size=training_config["batch_size"],
    )

    # Train based on model type
    model_type = config.get("model_type", "simple")
    mlflow_config = get_mlflow_config(config)

    # Add dataset metadata to MLflow config for logging
    mlflow_config["dataset_metadata"] = dataset_metadata

    if model_type == "simple":
        # Include weight_decay in optimizer_config
        optimizer_config = training_config["optimizer_config"] or {}
        optimizer_config["weight_decay"] = training_config["weight_decay"]

        trainer, model, _ = train_simple_encoder(
            data_module=data_module,
            out_neurons=config.get("out_neurons", 100),
            learning_rate=training_config["learning_rate"],
            epochs=training_config["epochs"],
            unfreeze_epoch=training_config["unfreeze_epoch"],
            optimizer_config=optimizer_config,
            **mlflow_config,
        )
    elif model_type == "skip":
        # Include weight_decay in optimizer_config
        optimizer_config = training_config["optimizer_config"] or {}
        optimizer_config["weight_decay"] = training_config["weight_decay"]

        trainer, model, _ = train_skip_connection_encoder(
            data_module=data_module,
            out_neurons=config.get("out_neurons", 100),
            learning_rate=training_config["learning_rate"],
            epochs=training_config["epochs"],
            unfreeze_epoch=training_config["unfreeze_epoch"],
            optimizer_config=optimizer_config,
            **mlflow_config,
        )
    elif model_type == "resnet":
        # Include weight_decay in optimizer_config
        optimizer_config = training_config["optimizer_config"] or {}
        optimizer_config["weight_decay"] = training_config["weight_decay"]

        trainer, model, _ = train_resnet_encoder(
            data_module=data_module,
            out_neurons=config.get("out_neurons", 100),
            resnet_type=config.get("resnet_type", "resnet18"),
            freeze_backbone=config.get("freeze_backbone", True),
            unfreeze_epoch=training_config["unfreeze_epoch"],
            learning_rate=training_config["learning_rate"],
            epochs=training_config["epochs"],
            optimizer_config=optimizer_config,
            **mlflow_config,
        )

    return trainer, model, data_module


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Generic MLflow Encoder Training"
    )

    # Model parameters
    parser.add_argument(
        "--model-type",
        default="simple",
        help="Model type (simple, skip, resnet)",
    )
    parser.add_argument(
        "--out-neurons", type=int, default=100, help="Number of output neurons"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--dataset-type",
        default="cifar10",
        help="Dataset type (cifar10, mnist)",
    )
    parser.add_argument(
        "--sta-type",
        default="perlin_noise_patterns,11,11",
        help="STA type for synthetic data",
    )
    parser.add_argument(
        "--n-neurons",
        type=int,
        default=1000,
        help="Number of neurons in synthetic data",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=1000,
        help="Number of images in synthetic data",
    )
    parser.add_argument(
        "--experiment-name",
        default="neural_encoder",
        help="MLflow experiment name",
    )
    parser.add_argument("--run-name", help="MLflow run name")
    parser.add_argument(
        "--array-task-id",
        type=int,
        help="SLURM array task ID for hyperparameter sweep",
    )

    args = parser.parse_args()

    # Check if this is a hyperparameter sweep
    if args.array_task_id is not None:
        # Hyperparameter sweep mode for SLURM job arrays
        # Define hyperparameter combinations
        learning_rates = [0.00001, 0.0001, 0.001, 0.01]
        batch_sizes = [8, 16, 32, 64]

        # Calculate which combination this array task should run
        lr_idx = args.array_task_id // len(batch_sizes)
        bs_idx = args.array_task_id % len(batch_sizes)

        if lr_idx >= len(learning_rates):
            raise ValueError(
                f"Array task ID {args.array_task_id} is out of range"
            )

        lr = learning_rates[lr_idx]
        batch_size = batch_sizes[bs_idx]
        run_name = f"resnet_lr{lr}_bs{batch_size}"

        print(
            f"Array Task {args.array_task_id}: lr={lr}, "
            f"batch_size={batch_size}"
        )

        config = {
            "model_type": "resnet",
            "out_neurons": 1000,
            "learning_rate": lr,
            "epochs": 10000,
            "batch_size": batch_size,
            "dataset_type": "mnist",
            "sta_type": "perlin_noise_patterns,11,11",
            "n_neurons": 1000,
            "n_images": 1000,
            "mlflow_experiment_name": "resnet_hyperparameter_sweep_mnist",
            "mlflow_run_name": run_name,
        }

        trainer, model, _ = train_with_config(config)

        print(f"\nTraining completed for {run_name}!")
        if model.train_losses:
            print(f"Final train loss: {model.train_losses[-1]:.4f}")
        if model.val_losses:
            print(f"Final validation loss: {model.val_losses[-1]:.4f}")

    else:
        # Single experiment with command line arguments
        config = {
            "model_type": args.model_type,
            "out_neurons": args.out_neurons,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "dataset_type": args.dataset_type,
            "sta_type": args.sta_type,
            "n_neurons": args.n_neurons,
            "n_images": args.n_images,
            "mlflow_experiment_name": args.experiment_name,
            "mlflow_run_name": args.run_name,
        }

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
