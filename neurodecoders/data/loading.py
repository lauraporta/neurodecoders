"""
Shared data loading and preprocessing utilities for encoder and decoder.
"""

import datetime
import os
from typing import Any, Dict, Tuple

import numpy as np

from neurodecoders.paths import get_path


def parse_dataset_metadata(filename: str) -> Dict[str, Any]:
    """
    Parse dataset metadata from synthetic data filename.

    Expected format:
    synthdata_dataset-{dataset_type}_sta-{sta_type}_n_neurons-{n}_n_images-{m}.npz
    """
    meta: Dict[str, Any] = {}
    try:
        name = filename.replace(".npz", "")
        if "dataset-" in name:
            meta["dataset_type"] = name.split("dataset-")[1].split("_")[0]
        if "sta-" in name:
            sta_part = name.split("sta-")[1].split("_n_neurons")[0]
            meta["sta_type"] = sta_part
            if "," in sta_part:
                parts = sta_part.split(",")
                meta["sta_pattern"] = parts[0]
                if len(parts) >= 3:
                    meta["sta_patch_width"] = str(int(parts[1]))
                    meta["sta_patch_height"] = str(int(parts[2]))
        if "n_neurons-" in name:
            meta["n_neurons"] = str(
                int(name.split("n_neurons-")[1].split("_")[0])
            )
        if "n_images-" in name:
            meta["n_images"] = str(
                int(name.split("n_images-")[1].split("_")[0])
            )
    except Exception:
        # Best-effort only
        pass
    return meta


def load_synthetic_split_data(
    config: Dict[str, Any],
    split: str = "train",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load synthetic data from workspace/datasets/synthetic split structure.
    
    Args:
        config: Configuration dictionary. Supports:
            - n_neurons: Number of neurons to match
            - n_images: Number of images (used if n_train_images/n_test_images not specified)
            - n_train_images: Number of training images (overrides n_images for train split)
            - n_test_images: Number of test images (overrides n_images for test split)
            - dataset_type: Dataset type (e.g., 'cifar10')
            - sta_type: STA type (e.g., 'gabor,100,100')
        split: Which split to load - "train" or "test" (default: "train")
    
    Returns:
        images, firing_rates, labels, and metadata.
        
    Example:
        # Load train data with 8000 images and test data with 2000 images
        config = {
            "n_neurons": 5000,
            "n_train_images": 8000,
            "n_test_images": 2000,
            "dataset_type": "cifar10",
            "sta_type": "gabor,11,11"
        }
        train_data = load_synthetic_split_data(config, split="train")
        test_data = load_synthetic_split_data(config, split="test")
    """
    if split not in ["train", "test"]:
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")
    
    synthetic_dir = get_path("workspace/datasets/synthetic")
    train_dir = os.path.join(synthetic_dir, "train")
    test_dir = os.path.join(synthetic_dir, "test")

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(
            "Split data directories not found. "
            "Please run the synthetic data generation first."
        )

    return _load_from_split_structure(config, synthetic_dir, split)


def _load_from_split_structure(
    config: Dict[str, Any], synthetic_dir: str, split: str = "train"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load data from specified split directory (train or test)."""
    split_dir = os.path.join(synthetic_dir, split)
    avail = [f for f in os.listdir(split_dir) if f.endswith(".npz")]
    if not avail:
        raise FileNotFoundError(
            f"No {split} data files found in {split_dir}. "
            "Please run the synthetic data generation first."
        )

    n_neurons = (
        str(int(config["n_neurons"])) if "n_neurons" in config else None
    )
    
    # Use split-specific n_images if provided, otherwise fall back to n_images
    n_images_key = f"n_{split}_images"
    if n_images_key in config and config[n_images_key] is not None:
        n_images = str(int(config[n_images_key]))
    elif "n_images" in config and config["n_images"] is not None:
        n_images = str(int(config["n_images"]))
    else:
        n_images = None
    
    sta_type = config.get("sta_type", "")
    dataset_type = config.get("dataset_type", "")

    candidates = []
    for fname in avail:
        meta = parse_dataset_metadata(fname)
        if (
            (n_neurons is None or meta.get("n_neurons") == n_neurons)
            and (n_images is None or meta.get("n_images") == n_images)
            and (not sta_type or meta.get("sta_type") == sta_type)
            and (not dataset_type or meta.get("dataset_type") == dataset_type)
        ):
            fpath = os.path.join(split_dir, fname)
            try:
                mtime = os.path.getmtime(fpath)
            except OSError:
                mtime = 0.0
            candidates.append((mtime, fname))

    if not candidates:
        raise ValueError(
            f"No {split} dataset matches the requested parameters. "
            f"Requested dataset_type={dataset_type or 'ANY'}, "
            f"n_neurons={n_neurons}, "
            f"n_images={n_images} (checked '{n_images_key}' and 'n_images'), "
            f"sta_type={sta_type}."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected_file = candidates[0][1]

    meta = parse_dataset_metadata(selected_file)
    meta["dataset_timestamp"] = datetime.datetime.now().isoformat()
    meta["dataset_filename"] = selected_file
    meta["data_split"] = split

    file_path = os.path.join(split_dir, selected_file)
    use_mmap = bool(config.get("use_memory_mapping"))
    data = np.load(file_path, mmap_mode="r" if use_mmap else None)

    if "images" not in data or (
        "responses" not in data and "firing_rates" not in data
    ):
        raise ValueError(
            "Invalid synthetic data format: missing 'images' or responses"
        )

    images = data["images"]
    firing = data.get("responses", data.get("firing_rates"))
    labels = data.get("labels", None)

    return images, firing, labels, meta


def load_dataset_path(dataset_to_load: str) -> str:
    """
    Resolve a dataset path from workspace for decoder use.
    """
    workspace_path = get_path("workspace")
    if os.path.isabs(dataset_to_load):
        path = dataset_to_load
    else:
        candidates = [
            os.path.join(workspace_path, dataset_to_load),
            os.path.join(
                workspace_path,
                "datasets",
                "synthetic",
                "train",
                dataset_to_load,
            ),
            os.path.join(workspace_path, "datasets", dataset_to_load),
        ]
        path = next((p for p in candidates if os.path.exists(p)), "")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found. Tried candidates under {workspace_path}."
        )
    return path


def load_npz_dataset(dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images and firing rates from an .npz file.
    """
    data = np.load(dataset_path)
    if "images" not in data:
        raise KeyError("No 'images' key found in dataset")
    images = data["images"]
    if "firing_rates" in data:
        firing = data["firing_rates"]
    elif "responses" in data:
        firing = data["responses"]
    else:
        raise KeyError("No 'firing_rates' or 'responses' key found in dataset")
    return images, firing


def normalize_images_and_rates(
    images: np.ndarray, firing_rates: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Normalize images to [-1, 1] and z-score firing rates per neuron.
    Returns images, firing_rates, H, W.
    """
    img_min, img_max = images.min(), images.max()
    if img_max == img_min:
        raise ValueError("Images have zero dynamic range")
    images = (images - img_min) / (img_max - img_min) * 2 - 1

    fr_std = firing_rates.std(axis=0)
    if np.any(fr_std == 0):
        raise ValueError("At least one neuron has zero-variance firing rates")
    firing_rates = (firing_rates - firing_rates.mean(axis=0)) / fr_std

    h, w = images.shape[1], images.shape[2]
    return images, firing_rates, h, w
