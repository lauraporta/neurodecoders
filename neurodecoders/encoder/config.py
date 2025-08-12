"""
Configuration defaults for neural encoder training.

This module centralizes all default values to ensure consistency across
the codebase and prevent parameter overwriting.
"""

from typing import Any, Dict

# Model configuration defaults
MODEL_DEFAULTS = {
    "model_type": "simple",
    "resnet_type": "resnet18",
    "freeze_backbone": True,
}

# Training configuration defaults
TRAINING_DEFAULTS = {
    "learning_rate": 0.001,
    "epochs": 30,
    "batch_size": 32,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "loss_function": "mse",
    "scheduler": "none",
    "scheduler_step_size": 30,
    "scheduler_gamma": 0.1,
}

# Data configuration defaults
DATA_DEFAULTS = {
    "dataset_type": "cifar10",
    "sta_type": "periodic_patterns,70,70",
    "n_neurons": 100,
    "n_images": 100000,
}

# Data loading configuration defaults
DATA_LOADING_DEFAULTS = {
    "use_memory_mapping": False,  # For very large datasets
    "chunk_size": 10000,  # Load data in chunks
    "prefetch_factor": 2,  # DataLoader prefetch
    "num_workers": 0,  # Number of workers for data loading
    "pin_memory": True,  # Pin memory for faster GPU transfer
}

# MLflow configuration defaults
MLFLOW_DEFAULTS = {
    "experiment_name": "neural_encoder",
    "enable_mlflow": True,
}

# Hyperparameter sweep defaults
HYPERPARAMETER_SWEEP_DEFAULTS = {
    "learning_rates": [0.00001, 0.0001, 0.001, 0.01],
    "batch_sizes": [8, 16, 32, 64],
    "sweep_epochs": 10000,
    "sweep_model_type": ["resnet", "simple", "skip"],
    "sweep_dataset_type": "cifar10",
}

# Supported options
SUPPORTED_MODEL_TYPES = ["simple", "skip", "resnet"]
SUPPORTED_RESNET_TYPES = ["resnet18", "resnet34", "resnet50"]
SUPPORTED_OPTIMIZERS = ["adam", "adamw", "sgd"]
SUPPORTED_LOSS_FUNCTIONS = ["mse", "l1", "smooth_l1", "huber"]
SUPPORTED_SCHEDULERS = ["none", "step", "cosine", "plateau"]


def get_default_config() -> Dict[str, Any]:
    """
    Get a complete default configuration dictionary.

    Returns:
        Dictionary with all default values
    """
    config = {}
    config.update(MODEL_DEFAULTS)
    config.update(TRAINING_DEFAULTS)
    config.update(DATA_DEFAULTS)
    config.update(DATA_LOADING_DEFAULTS)
    config.update(MLFLOW_DEFAULTS)
    return config


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


def merge_config_with_defaults(user_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge user configuration with defaults.

    Args:
        user_config: User-provided configuration

    Returns:
        Complete configuration with defaults filled in
    """
    default_config = get_default_config()
    merged_config = default_config.copy()
    merged_config.update(user_config)
    return merged_config
