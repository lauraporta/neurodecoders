#!/usr/bin/env python3
"""
Generic MLflow training script for neural encoders.

This script provides a flexible interface for training different encoder models
with various configurations, datasets, and hyperparameters while tracking
experiments with MLflow.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

# Add the encoder directory to the path
sys.path.append(os.path.dirname(__file__))

import numpy as np
import torch

from neurodecoders.encoder.mlflow_utils import get_experiment_comparison
from neurodecoders.encoder.models import (
    ResNetEncoder,
    SimpleEncoder,
    SimpleEncoderWithSkipConnection,
)
from neurodecoders.encoder.training import (
    train_encoder,
    train_resnet_encoder,
    train_simple_encoder,
)
from neurodecoders.encoder.utils import NeuralDataModule, preprocess_data


def load_synthetic_data_from_workspace(config: Dict[str, Any]) -> tuple:
    """
    Load synthetic data from workspace/datasets/synthetic based on
    configuration.

    Args:
        config: Configuration dictionary with data parameters

    Returns:
        images, firing_rates: Synthetic data loaded from workspace
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

    # Load the data
    file_path = os.path.join(synthetic_dir, selected_file)
    print(f"Loading synthetic data from: {file_path}")

    try:
        data = np.load(file_path)

        # Extract images and responses (firing rates)
        if "images" in data and "responses" in data:
            images = data["images"]
            firing_rates = data["responses"]
        else:
            raise ValueError(
                "Invalid synthetic data format: missing 'images' or "
                "'responses'"
            )

        print(
            f"Loaded data: {images.shape} images, "
            f"{firing_rates.shape} firing rates"
        )

        return images, firing_rates

    except Exception as e:
        raise RuntimeError(
            f"Error loading synthetic data from {file_path}: {e}"
        )


def load_real_data(config: Dict[str, Any]) -> tuple:
    """
    Load real data based on configuration.

    Args:
        config: Configuration dictionary with data parameters

    Returns:
        images, firing_rates: Real data
    """
    data_path = config.get("data_path")
    if not data_path or not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Real data path {data_path} not found. "
            "Please provide a valid path to real data or use synthetic data "
            "from workspace."
        )

    # Load data from file
    try:
        data = np.load(data_path)
        if "images" in data and "firing_rates" in data:
            return data["images"], data["firing_rates"]
        else:
            raise ValueError(
                "Invalid data file format: missing 'images' or 'firing_rates'"
            )
    except Exception as e:
        raise RuntimeError(f"Error loading real data from {data_path}: {e}")


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
    images, firing_rates = load_synthetic_data_from_workspace(config)

    # Ensure data matches model configuration
    out_neurons = config.get("out_neurons", 100)
    if firing_rates.shape[1] != out_neurons:
        print(
            f"Adjusting firing rates from {firing_rates.shape[1]} to "
            f"{out_neurons} neurons"
        )
        if firing_rates.shape[1] > out_neurons:
            # Truncate to match model
            firing_rates = firing_rates[:, :out_neurons]
        else:
            # Pad with zeros to match model
            padding = np.zeros(
                (firing_rates.shape[0], out_neurons - firing_rates.shape[1])
            )
            firing_rates = np.hstack([firing_rates, padding])

    # Preprocess data
    images, firing_rates = preprocess_data(images, firing_rates)

    # Create data module
    training_config = get_training_config(config)
    data_module = NeuralDataModule(
        images=images,
        firing_rates=firing_rates,
        train_split=training_config["train_split"],
        val_split=training_config["val_split"],
        batch_size=training_config["batch_size"],
    )

    # Train based on model type
    model_type = config.get("model_type", "simple")
    mlflow_config = get_mlflow_config(config)

    if model_type == "simple":
        trainer, model, _ = train_simple_encoder(
            data_module=data_module,
            out_neurons=config.get("out_neurons", 100),
            learning_rate=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"],
            epochs=training_config["epochs"],
            unfreeze_epoch=training_config["unfreeze_epoch"],
            optimizer_config=training_config["optimizer_config"],
            **mlflow_config,
        )
    elif model_type == "resnet":
        trainer, model, _ = train_resnet_encoder(
            data_module=data_module,
            out_neurons=config.get("out_neurons", 100),
            resnet_type=config.get("resnet_type", "resnet18"),
            freeze_backbone=config.get("freeze_backbone", True),
            unfreeze_epoch=training_config["unfreeze_epoch"],
            learning_rate=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"],
            epochs=training_config["epochs"],
            optimizer_config=training_config["optimizer_config"],
            **mlflow_config,
        )
    else:
        # Generic training for custom models
        model = get_model(config)
        trainer, lightning_model, _ = train_encoder(
            model=model,
            data_module=data_module,
            model_name=f"{model_type}_encoder",
            logger_name=f"{model_type}_encoder",
            **training_config,
            **mlflow_config,
        )
        model = lightning_model

    return trainer, model, data_module


def run_experiment_comparison(configs: list) -> None:
    """
    Run multiple experiments with different configurations.

    Args:
        configs: List of configuration dictionaries
    """
    print(f"\n=== Running {len(configs)} experiments ===")

    results = []
    for i, config in enumerate(configs):
        print(f"\n--- Experiment {i + 1}/{len(configs)} ---")

        # Generate run name if not provided
        if not config.get("mlflow_run_name"):
            config["mlflow_run_name"] = (
                f"{config.get('model_type', 'model')}_{i + 1}"
            )

        try:
            trainer, model, data_module = train_with_config(config)

            # Store results
            result = {
                "config": config,
                "trainer": trainer,
                "model": model,
                "final_train_loss": model.train_losses[-1]
                if model.train_losses
                else None,
                "final_val_loss": model.val_losses[-1]
                if model.val_losses
                else None,
            }
            results.append(result)

            print(f"✓ Experiment {i + 1} completed successfully")

        except Exception as e:
            print(f"✗ Experiment {i + 1} failed: {e}")

    # Compare results
    if results and configs[0].get("enable_mlflow", True):
        experiment_name = configs[0].get(
            "mlflow_experiment_name", "neural_encoder"
        )
        print("\n=== Experiment Comparison ===")
        comparison_df = get_experiment_comparison(experiment_name)

        if not comparison_df.empty:
            print("Experiment Results:")
            print(
                comparison_df[
                    [
                        "run_name",
                        "params.learning_rate",
                        "params.epochs",
                        "metrics.final_train_loss",
                        "metrics.final_val_loss",
                    ]
                ].to_string()
            )
        else:
            print("No experiment data found.")


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        config: Configuration dictionary
    """
    # Handle relative paths from root directory
    if not os.path.isabs(config_path):
        # Try different possible locations
        possible_paths = [
            config_path,  # As provided
            os.path.join("neurodecoders", "encoder", config_path),  # From root
            os.path.join(
                "neurodecoders", "encoder", "configs", config_path
            ),  # From root with configs
            os.path.join(
                os.path.dirname(__file__), config_path
            ),  # Relative to script
            os.path.join(
                os.path.dirname(__file__), "configs", config_path
            ),  # Relative to script with configs
        ]

        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        else:
            raise FileNotFoundError(
                f"Config file not found. Tried: {possible_paths}"
            )

    with open(config_path, "r") as f:
        return json.load(f)


def create_sample_configs() -> list:
    """
    Create sample configurations for demonstration.

    Returns:
        configs: List of sample configurations
    """
    return [
        # Simple encoder with default parameters
        {
            "model_type": "simple",
            "out_neurons": 50,
            "learning_rate": 1e-3,
            "epochs": 10,
            "dataset_type": "cifar10",
            "sta_type": "perlin_noise_patterns,11,11",
            "n_neurons": 1000,
            "n_images": 1000,
            "mlflow_experiment_name": "encoder_comparison",
            "mlflow_run_name": "simple_default",
        },
        # Simple encoder with high learning rate
        {
            "model_type": "simple",
            "out_neurons": 50,
            "learning_rate": 1e-2,
            "epochs": 10,
            "dataset_type": "cifar10",
            "sta_type": "perlin_noise_patterns,11,11",
            "n_neurons": 1000,
            "n_images": 1000,
            "mlflow_experiment_name": "encoder_comparison",
            "mlflow_run_name": "simple_high_lr",
        },
        # ResNet encoder
        {
            "model_type": "resnet",
            "resnet_type": "resnet18",
            "out_neurons": 50,
            "freeze_backbone": True,
            "unfreeze_epoch": 5,
            "epochs": 10,
            "dataset_type": "cifar10",
            "sta_type": "perlin_noise_patterns,11,11",
            "n_neurons": 1000,
            "n_images": 1000,
            "mlflow_experiment_name": "encoder_comparison",
            "mlflow_run_name": "resnet_encoder",
        },
    ]


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Generic MLflow Encoder Training"
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["single", "comparison", "config"],
        default="single",
        help="Training mode: single experiment, comparison, or config file",
    )

    # Configuration file
    parser.add_argument(
        "--config", type=str, help="Path to JSON configuration file"
    )

    # Quick parameters for single mode
    parser.add_argument(
        "--model-type", default="simple", help="Model type (simple, resnet)"
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

    args = parser.parse_args()

    if args.mode == "config" and args.config:
        # Load configuration from file
        config = load_config_from_file(args.config)
        if isinstance(config, list):
            run_experiment_comparison(config)
        else:
            trainer, model, _ = train_with_config(config)

    elif args.mode == "comparison":
        # Run comparison with sample configs
        configs = create_sample_configs()
        run_experiment_comparison(configs)

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
