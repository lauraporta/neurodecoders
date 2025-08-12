"""
Path configuration utilities for neurodecoders.

This module provides functions to manage all artifact paths based on
a single base path configuration.
"""

import os
from pathlib import Path

import yaml


def get_base_path() -> str:
    """
    Get the base path from config.yaml.

    Returns:
        Base path string from config file
    """
    config_path = "config.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config.get("base_path", ".")
    except FileNotFoundError:
        print(f"Warning: {config_path} not found, using current directory")
        return "."
    except yaml.YAMLError as e:
        print(
            f"Warning: Error parsing {config_path}: {e}, "
            "using current directory"
        )
        return "."


def get_path(relative_path: str) -> str:
    """
    Get absolute path by combining base path with relative path.

    Args:
        relative_path: Relative path from base directory

    Returns:
        Absolute path
    """
    base = get_base_path()
    return os.path.join(base, relative_path)


def ensure_dir(path: str) -> None:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path to ensure exists
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def get_workspace_path() -> str:
    """
    Get the workspace directory path.

    Returns:
        Path to workspace directory
    """
    return get_path("workspace")


def get_mlflow_path() -> str:
    """
    Get the MLflow directory path.

    Returns:
        Path to MLflow directory
    """
    return get_path("mlruns")


def get_synthetic_data_path() -> str:
    """
    Get the synthetic data directory path.

    Returns:
        Path to synthetic data directory
    """
    return get_path("workspace/datasets/synthetic")


def get_encoder_models_path() -> str:
    """
    Get the encoder models directory path.

    Returns:
        Path to encoder models directory
    """
    return get_path("workspace/models/encoders")


def get_decoder_models_path() -> str:
    """
    Get the decoder models directory path.

    Returns:
        Path to decoder models directory
    """
    return get_path("workspace/models/decoders")


def get_plots_path() -> str:
    """
    Get the plots directory path.

    Returns:
        Path to plots directory
    """
    return get_path("workspace/plots")


def get_predictions_path() -> str:
    """
    Get the predictions directory path.

    Returns:
        Path to predictions directory
    """
    return get_path("workspace/predictions")
