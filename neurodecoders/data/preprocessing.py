"""
Data preprocessing utilities for neurodecoders.

This module provides normalization and preprocessing functions that can be
shared across encoder and decoder training pipelines.

For convenience, the main normalization functions are re-exported from
data.loading where they are currently defined.
"""

from typing import Dict, Tuple

import numpy as np

# Re-export normalization functions from loading module
from neurodecoders.data.loading import (
    apply_normalization,
    compute_and_apply_normalization,
    compute_normalization_stats,
)

__all__ = [
    "apply_normalization",
    "compute_and_apply_normalization",
    "compute_normalization_stats",
    "normalize_images_zscore",
    "normalize_firing_rates_zscore",
    "denormalize_images",
]


def normalize_images_zscore(
    images: np.ndarray,
    mean: float = None,
    std: float = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Z-score normalize images.

    Args:
        images: Images array of any shape
        mean: Pre-computed mean (if None, computed from images)
        std: Pre-computed std (if None, computed from images)

    Returns:
        Normalized images, mean used, std used
    """
    if mean is None:
        mean = float(images.mean())
    if std is None:
        std = float(images.std())

    if std == 0:
        std = 1.0

    normalized = (images - mean) / std
    return normalized, mean, std


def normalize_firing_rates_zscore(
    firing_rates: np.ndarray,
    mean: np.ndarray = None,
    std: np.ndarray = None,
    axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalize firing rates (typically per-neuron).

    Args:
        firing_rates: Firing rates array (N, n_neurons) typically
        mean: Pre-computed mean per neuron (if None, computed from data)
        std: Pre-computed std per neuron (if None, computed from data)
        axis: Axis along which to compute statistics (0 = per neuron)

    Returns:
        Normalized firing rates, mean used, std used
    """
    if mean is None:
        mean = firing_rates.mean(axis=axis)
    if std is None:
        std = firing_rates.std(axis=axis)

    # Handle zero std by setting to 1.0
    std = np.where(std == 0, 1.0, std)

    normalized = (firing_rates - mean) / std
    return normalized, mean, std


def denormalize_images(
    normalized_images: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """
    Reverse z-score normalization for images.

    Args:
        normalized_images: Normalized images
        mean: Mean used for normalization
        std: Std used for normalization

    Returns:
        Denormalized images
    """
    return normalized_images * std + mean


def clip_to_valid_range(
    images: np.ndarray,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> np.ndarray:
    """
    Clip images to a valid range.

    Args:
        images: Images to clip
        min_val: Minimum valid value
        max_val: Maximum valid value

    Returns:
        Clipped images
    """
    return np.clip(images, min_val, max_val)


def get_normalization_stats_summary(stats: Dict[str, float]) -> str:
    """
    Get a human-readable summary of normalization statistics.

    Args:
        stats: Dictionary with normalization statistics

    Returns:
        Formatted string summary
    """
    lines = [
        "Normalization Statistics:",
        f"  Image mean: {stats.get('image_mean', 'N/A'):.4f}",
        f"  Image std: {stats.get('image_std', 'N/A'):.4f}",
    ]

    firing_mean = stats.get("firing_mean")
    firing_std = stats.get("firing_std")

    if firing_mean is not None:
        if isinstance(firing_mean, (list, np.ndarray)):
            lines.append(f"  Firing mean: per-neuron, range [{min(firing_mean):.4f}, {max(firing_mean):.4f}]")
        else:
            lines.append(f"  Firing mean: {firing_mean:.4f}")

    if firing_std is not None:
        if isinstance(firing_std, (list, np.ndarray)):
            lines.append(f"  Firing std: per-neuron, range [{min(firing_std):.4f}, {max(firing_std):.4f}]")
        else:
            lines.append(f"  Firing std: {firing_std:.4f}")

    return "\n".join(lines)
