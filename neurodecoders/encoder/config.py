"""
Configuration validation for neural encoder training.

This module provides validation functions and supported options to ensure
consistency across the codebase.
"""

from typing import Any, Dict

# Hyperparameter sweep defaults (used for SLURM job arrays)
HYPERPARAMETER_SWEEP_DEFAULTS = {
    "learning_rates": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],  # More granular
    "batch_sizes": [8, 16, 32, 64, 128],  # Larger batch sizes for efficiency
    "sweep_epochs": 2,  # Set to 2 epochs for testing
    "sweep_model_type": ["resnet", "simple", "skip"],  # Model comparison
    "sweep_dataset_type": "cifar10",
    "optimizers": ["adam"],  # Optimizer comparison
    "schedulers": ["none"],  # Learning rate scheduling
    "loss_functions": ["mse", "smooth_l1"],  # Loss function comparison
}

# Supported options
SUPPORTED_MODEL_TYPES = ["simple", "skip", "resnet"]
SUPPORTED_RESNET_TYPES = ["resnet18", "resnet34", "resnet50"]
SUPPORTED_OPTIMIZERS = ["adam", "adamw", "sgd"]
SUPPORTED_LOSS_FUNCTIONS = ["mse", "l1", "smooth_l1", "huber"]
SUPPORTED_SCHEDULERS = ["none", "step", "cosine", "plateau"]


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration parameters.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If any parameter is invalid
    """
    # Validate model type
    if "model_type" in config:
        if config["model_type"] not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type: {config['model_type']}. "
                f"Supported: {SUPPORTED_MODEL_TYPES}"
            )

    # Validate ResNet type
    if "resnet_type" in config:
        if config["resnet_type"] not in SUPPORTED_RESNET_TYPES:
            raise ValueError(
                f"Unsupported resnet_type: {config['resnet_type']}. "
                f"Supported: {SUPPORTED_RESNET_TYPES}"
            )

    # Validate optimizer
    if "optimizer" in config:
        if config["optimizer"] not in SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"Unsupported optimizer: {config['optimizer']}. "
                f"Supported: {SUPPORTED_OPTIMIZERS}"
            )

    # Validate loss function
    if "loss_function" in config:
        if config["loss_function"] not in SUPPORTED_LOSS_FUNCTIONS:
            raise ValueError(
                f"Unsupported loss_function: {config['loss_function']}. "
                f"Supported: {SUPPORTED_LOSS_FUNCTIONS}"
            )

    # Validate scheduler
    if "scheduler" in config:
        if config["scheduler"] not in SUPPORTED_SCHEDULERS:
            raise ValueError(
                f"Unsupported scheduler: {config['scheduler']}. "
                f"Supported: {SUPPORTED_SCHEDULERS}"
            )

    # Validate numeric parameters
    if "learning_rate" in config and config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")

    if "epochs" in config and config["epochs"] <= 0:
        raise ValueError("epochs must be positive")

    if "batch_size" in config and config["batch_size"] <= 0:
        raise ValueError("batch_size must be positive")

    if "weight_decay" in config and config["weight_decay"] < 0:
        raise ValueError("weight_decay must be non-negative")

    if "n_neurons" in config and config["n_neurons"] <= 0:
        raise ValueError("n_neurons must be positive")

    if "n_images" in config and config["n_images"] <= 0:
        raise ValueError("n_images must be positive")
