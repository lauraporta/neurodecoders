import argparse
import datetime
import os

import numpy as np
import torch
from image_datasets import ImageDataset
from scipy.stats import pearsonr
from simulate_response import SimulateResponse
from sta import STA

from neurodecoders.paths import get_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_sta_vs_zscore_correlations(images, responses, stas, rf_coords):
    """
    Calculate correlations between STAs and corresponding patches from
    z-score normalized average images.

    Args:
        images (torch.Tensor): Image tensor of shape (n_images, channels,
            height, width)
        responses (np.ndarray): Firing rates of shape (n_images, n_neurons)
        stas (np.ndarray): STAs of shape (n_neurons, height, width) or
            (n_neurons, channels, height, width)
        rf_coords (np.ndarray): Receptive field coordinates of shape
            (n_neurons, 2)

    Returns:
        np.ndarray: Correlations for each neuron
    """
    print("Calculating STA vs z-score average image correlations...")

    # Convert images to numpy if needed
    if isinstance(images, torch.Tensor):
        images_np = images.cpu().numpy()
    else:
        images_np = images

    n_neurons = responses.shape[1]
    sta_avg_correlations = np.zeros(n_neurons)

    for neuron_idx in range(n_neurons):
        # Get firing rates for this neuron
        neuron_responses = responses[:, neuron_idx]

        # Z-score normalization: (fr - mean) / std
        mean_fr = np.mean(neuron_responses)
        std_fr = np.std(neuron_responses)

        if std_fr > 0:
            # Z-score normalization
            weights = (neuron_responses - mean_fr) / std_fr
        else:
            # If std is 0, use uniform weights
            weights = np.ones_like(neuron_responses) / len(neuron_responses)

        # Compute weighted average image for this neuron
        weights_reshaped = weights.reshape(-1, 1, 1, 1)
        weighted_images = images_np * weights_reshaped
        avg_img = np.sum(weighted_images, axis=0)

        # Get STA and patch from average image
        sta = stas[neuron_idx]
        x, y = rf_coords[neuron_idx]

        # Get the patch from the average image corresponding
        # to the STA location
        if len(avg_img.shape) == 3:
            # Take first channel if 3D
            patch = avg_img[0, y : y + sta.shape[0], x : x + sta.shape[1]]
        else:
            patch = avg_img[y : y + sta.shape[0], x : x + sta.shape[1]]

        # Flatten both to 1D arrays
        sta_flat = sta.flatten()
        patch_flat = patch.flatten()

        # Ensure both arrays have the same length
        min_length = min(len(sta_flat), len(patch_flat))
        sta_flat = sta_flat[:min_length]
        patch_flat = patch_flat[:min_length]

        # Compute correlation
        correlation, _ = pearsonr(sta_flat, patch_flat)
        sta_avg_correlations[neuron_idx] = correlation

        if (neuron_idx + 1) % 100 == 0:
            print(f"Processed {neuron_idx + 1}/{n_neurons} neurons")

    return sta_avg_correlations


def split_dataset(
    images,
    responses,
    stas,
    coords,
    adaptation_states,
    labels,
    train_split=0.8,
    random_seed=42,
):
    """
    Split dataset into train and test sets.

    Args:
        images: Image tensor
        responses: Firing rates array
        stas: STA patterns array
        coords: RF coordinates array
        adaptation_states: Adaptation states array
        labels: Labels tensor
        train_split: Fraction for training (default: 0.8)
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with train/test splits
    """
    n_samples = len(images)
    indices = np.arange(n_samples)

    # Set random seed for reproducibility
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    # Calculate split sizes
    train_size = int(n_samples * train_split)
    test_size = n_samples - train_size

    # Split indices
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    print(f"Dataset split: {train_size} train, {test_size} test")

    # Create splits
    splits = {}
    for split_name, split_indices in [
        ("train", train_indices),
        ("test", test_indices),
    ]:
        splits[split_name] = {
            "images": images[split_indices],
            "responses": responses[split_indices],
            "stas": stas,  # STAs are the same for all splits
            "rf_coords": coords,  # RF coords are the same for all splits
            "adaptation_states": adaptation_states[split_indices],
            "labels": labels[split_indices] if labels is not None else None,
        }

    return splits


def save_output(
    images,
    responses,
    stas,
    coords,
    adaptation_states,
    labels,
    dataset_type,
    sta_type,
    n_neurons,
    n_images,
    sta_avg_correlations=None,
    train_split=0.8,
):
    """
    Save dataset with train and test splits in separate folders.

    Args:
        images: Image tensor
        responses: Firing rates array
        stas: STA patterns array
        coords: RF coordinates array
        adaptation_states: Adaptation states array
        labels: Labels tensor
        dataset_type: Type of dataset
        sta_type: Type of STA patterns
        n_neurons: Number of neurons
        n_images: Number of images
        sta_avg_correlations: STA correlations array
        train_split: Fraction for training
    """
    # Split the dataset
    splits = split_dataset(
        images,
        responses,
        stas,
        coords,
        adaptation_states,
        labels,
        train_split=train_split,
    )

    # Create base output directory
    base_output_dir = get_path("workspace/datasets/synthetic")
    os.makedirs(base_output_dir, exist_ok=True)

    # Generate timestamp for consistent naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save each split
    for split_name, split_data in splits.items():
        # Create split-specific directory
        split_dir = os.path.join(base_output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # Generate filename for this split
        filename = (
            f"synthdata_dataset-{dataset_type}_sta-{sta_type}_n_neurons-"
            f"{n_neurons}_n_images-{len(split_data['images'])}_split-{split_name}_"
            f"datetime-{timestamp}.npz"
        )
        filepath = os.path.join(split_dir, filename)

        # Prepare save data
        save_data = {
            "images": split_data["images"].cpu().numpy(),
            "responses": split_data["responses"],
            "stas": split_data["stas"],
            "rf_coords": split_data["rf_coords"],
            "adaptation_states": split_data["adaptation_states"],
            "labels": split_data["labels"].cpu().numpy()
            if split_data["labels"] is not None
            else None,
        }

        # Add correlations if provided (same for all splits)
        if sta_avg_correlations is not None:
            save_data["sta_avg_correlations"] = sta_avg_correlations

        # Save the split
        np.savez(filepath, **save_data)
        print(f"Saved {split_name} split: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Create synthetic neural responses "
        "with configurable parameters."
    )
    parser.add_argument(
        "--n_images",
        type=int,
        default=100,
        help="Number of images to generate responses for (default: 100)",
    )
    parser.add_argument(
        "--n_neurons",
        type=int,
        default=100,
        help="Number of neurons to simulate (default: 100)",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="cifar10",
        help="Dataset type to use (default: cifar10)",
    )
    parser.add_argument(
        "--sta_type",
        type=str,
        default="periodic_patterns,70,70",
        help="STA type and parameters (default: periodic_patterns,70,70)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for memory-efficient processing (default: 100)",
    )

    args = parser.parse_args()

    print("Configuration:")
    print(f"  Dataset: {args.dataset_type}")
    print(f"  STA type: {args.sta_type}")
    print(f"  Number of images: {args.n_images}")
    print(f"  Number of neurons: {args.n_neurons}")
    print(f"  Batch size: {args.batch_size}")
    print()

    print("Loading data and model...")
    images, labels = ImageDataset().get_data(
        args.dataset_type, n_images=args.n_images
    )
    stas = STA().get_simulated_sta(args.sta_type)

    print("Generating responses...")
    simulator = SimulateResponse(device, images, stas, args.n_neurons)

    # Show memory estimates
    memory_info = simulator.estimate_memory_usage(args.batch_size)
    print(f"Memory estimates for batch size {args.batch_size}:")
    print(f"  Patch memory: {memory_info['patch_memory_gb']:.2f} GB")
    print(f"  Other tensors: {memory_info['other_tensors_gb']:.2f} GB")
    print(f"  Total memory: {memory_info['total_memory_gb']:.2f} GB")
    print(f"  Suggested batch size: {memory_info['suggested_batch_size']}")
    print()

    firing_rates, dot_products, noise = (
        simulator.simulate_neural_responses_vectorized(
            batch_size=args.batch_size
        )
    )

    # Create output directory
    os.makedirs("output", exist_ok=True)

    print("Saving dataset...")
    # Skip STA vs z-score correlation calculation for speed
    sta_avg_correlations = None

    save_output(
        images,
        firing_rates,
        simulator.selected_stas,
        simulator.rf_coords,
        noise,
        labels,
        args.dataset_type,
        args.sta_type,
        args.n_neurons,
        args.n_images,
        sta_avg_correlations,
        train_split=0.8,
    )
    print("Done.")


if __name__ == "__main__":
    main()
