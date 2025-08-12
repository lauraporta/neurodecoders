import argparse
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from image_datasets import ImageDataset
from matplotlib.patches import Rectangle
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
):
    output_dir = get_path("workspace/datasets/synthetic")
    os.makedirs(output_dir, exist_ok=True)
    filename = (
        f"synthdata_dataset-{dataset_type}_sta-{sta_type}_n_neurons-"
        f"{n_neurons}_n_images-{n_images}_datetime-"
        f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    )
    filepath = os.path.join(output_dir, filename)

    # Prepare save data
    save_data = {
        "images": images.cpu().numpy(),
        "responses": responses,
        "stas": stas,
        "rf_coords": coords,
        "adaptation_states": adaptation_states,
        "labels": labels.cpu().numpy(),
    }

    # Add correlations if provided
    if sta_avg_correlations is not None:
        save_data["sta_avg_correlations"] = sta_avg_correlations

    np.savez(filepath, **save_data)


def plot_sta_and_spikes(
    images,
    responses,
    dot_products,
    adaptation_states,
    stas,
    coords,
    n_plot_images=5,
    n_top_neurons=5,
):
    # Infer RF size from STA data
    if len(stas.shape) == 3:
        rf_size = stas.shape[1]  # (n_neurons, height, width)
    else:
        rf_size = stas.shape[2]  # (n_neurons, channels, height, width)

    print(f"Inferred RF size from STA data: {rf_size}x{rf_size}")

    # First sort images by their maximum firing rate
    image_max_responses = np.max(responses, axis=1)
    image_sort_idx = np.argsort(image_max_responses)[::-1]  # Descending order
    # Convert tensor to numpy for sorting
    images_np = images.cpu().numpy()
    sorted_images = torch.from_numpy(images_np[image_sort_idx]).to(
        images.device
    )
    sorted_responses = responses[image_sort_idx]
    sorted_dot_products = dot_products[image_sort_idx]
    sorted_adaptation = adaptation_states[image_sort_idx]

    # Then sort neurons by their response to the highest responding image
    neuron_sort_idx = np.argsort(sorted_responses[0])[::-1]  # Descending order
    # Take only top n_top_neurons
    neuron_sort_idx = neuron_sort_idx[:n_top_neurons]
    sorted_responses = sorted_responses[:, neuron_sort_idx]
    sorted_dot_products = sorted_dot_products[:, neuron_sort_idx]
    sorted_adaptation = sorted_adaptation[:, neuron_sort_idx]
    sorted_stas = stas[neuron_sort_idx]
    sorted_coords = coords[neuron_sort_idx]

    # Find global max for y-axis scaling
    y_max = np.max(sorted_responses)  # Only use firing rates for y_max now

    # Create first figure for images and responses
    fig1 = plt.figure(figsize=(30, 20))  # Increased from (20, 15)
    gs = fig1.add_gridspec(
        n_plot_images, 9
    )  # 9 columns: 4 for first set, 4 for second set, 1 for spacing

    # Colors for different neurons
    colors = plt.cm.tab10(np.linspace(0, 1, n_top_neurons))

    def remove_top_right_spines(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Plot responses for each image in first panel (left side)
    for i in range(n_plot_images):
        # First set of images
        # Plot image with receptive fields
        ax_img = fig1.add_subplot(gs[i, 0])
        img_raw = sorted_images[i][0].cpu().numpy()
        # Convert from [-1, 1] to [0, 1] for display
        img_raw = (img_raw + 1) / 2
        img_raw = np.clip(img_raw, 0, 1)
        ax_img.imshow(img_raw, cmap="gray")

        # Add receptive field rectangles for all neurons
        for n in range(n_top_neurons):
            x, y = sorted_coords[n]
            rect = Rectangle(
                (x, y),
                rf_size,
                rf_size,
                linewidth=3.0,
                edgecolor=colors[n],
                facecolor="none",
                alpha=0.8,
            )
            ax_img.add_patch(rect)
        ax_img.axis("off")
        ax_img.set_title(
            f"Image {image_sort_idx[i]} (Max Response: "
            f"{image_max_responses[image_sort_idx[i]]:.2f})"
        )

        # Plot firing rates
        ax_fr = fig1.add_subplot(gs[i, 1])
        ax_fr.scatter(
            range(n_top_neurons),
            sorted_responses[i],
            c=colors[:n_top_neurons],
            s=100,
        )
        ax_fr.set_title("Firing Rates")
        ax_fr.set_xlabel("Neuron")
        ax_fr.set_ylabel("Rate (Hz)")
        ax_fr.set_xticks(range(n_top_neurons))
        ax_fr.set_ylim(0, y_max)
        remove_top_right_spines(ax_fr)

        # Plot dot products
        ax_dp = fig1.add_subplot(gs[i, 2])
        ax_dp.scatter(
            range(n_top_neurons),
            sorted_dot_products[i],
            c=colors[:n_top_neurons],
            s=100,
            marker="x",
        )
        ax_dp.set_title("Dot Products")
        ax_dp.set_xlabel("Neuron")
        ax_dp.set_ylabel("Dot Product")
        ax_dp.set_xticks(range(n_top_neurons))
        ax_dp.set_ylim(0, 1)  # Dot products are normalized to [0,1]
        remove_top_right_spines(ax_dp)

        # Plot adaptation state
        ax_ad = fig1.add_subplot(gs[i, 3])
        ax_ad.scatter(
            range(n_top_neurons),
            sorted_adaptation[i],
            c=colors[:n_top_neurons],
            s=100,
            marker="s",
        )
        ax_ad.set_title("Adaptation State")
        ax_ad.set_xlabel("Neuron")
        ax_ad.set_ylabel("Adaptation")
        ax_ad.set_xticks(range(n_top_neurons))
        ax_ad.set_ylim(0, 1)  # Adaptation state is between 0 and 1
        remove_top_right_spines(ax_ad)

        # Second set of images
        # Plot image with receptive fields
        ax_img2 = fig1.add_subplot(gs[i, 5])
        img_raw = sorted_images[i + n_plot_images][0].cpu().numpy()
        img_raw = (img_raw + 1) / 2
        img_raw = np.clip(img_raw, 0, 1)
        ax_img2.imshow(img_raw, cmap="gray")

        # Add receptive field rectangles for all neurons
        for n in range(n_top_neurons):
            x, y = sorted_coords[n]
            rect = Rectangle(
                (x, y),
                rf_size,
                rf_size,
                linewidth=3.0,
                edgecolor=colors[n],
                facecolor="none",
                alpha=0.8,
            )
            ax_img2.add_patch(rect)
        ax_img2.axis("off")
        ax_img2.set_title(
            f"Image {image_sort_idx[i + n_plot_images]} (Max Response: "
            f"{image_max_responses[image_sort_idx[i + n_plot_images]]:.2f})"
        )

        # Plot firing rates for second set
        ax_fr2 = fig1.add_subplot(gs[i, 6])
        ax_fr2.scatter(
            range(n_top_neurons),
            sorted_responses[i + n_plot_images],
            c=colors[:n_top_neurons],
            s=100,
        )
        ax_fr2.set_title("Firing Rates")
        ax_fr2.set_xlabel("Neuron")
        ax_fr2.set_ylabel("Rate (Hz)")
        ax_fr2.set_xticks(range(n_top_neurons))
        ax_fr2.set_ylim(0, y_max)
        remove_top_right_spines(ax_fr2)

        # Plot dot products for second set
        ax_dp2 = fig1.add_subplot(gs[i, 7])
        ax_dp2.scatter(
            range(n_top_neurons),
            sorted_dot_products[i + n_plot_images],
            c=colors[:n_top_neurons],
            s=100,
            marker="x",
        )
        ax_dp2.set_title("Dot Products")
        ax_dp2.set_xlabel("Neuron")
        ax_dp2.set_ylabel("Dot Product")
        ax_dp2.set_xticks(range(n_top_neurons))
        ax_dp2.set_ylim(0, 1)  # Dot products are normalized to [0,1]
        remove_top_right_spines(ax_dp2)

        # Plot adaptation state for second set
        ax_ad2 = fig1.add_subplot(gs[i, 8])
        ax_ad2.scatter(
            range(n_top_neurons),
            sorted_adaptation[i + n_plot_images],
            c=colors[:n_top_neurons],
            s=100,
            marker="s",
        )
        ax_ad2.set_title("Adaptation State")
        ax_ad2.set_xlabel("Neuron")
        ax_ad2.set_ylabel("Adaptation")
        ax_ad2.set_xticks(range(n_top_neurons))
        ax_ad2.set_ylim(0, 1)  # Adaptation state is between 0 and 1
        remove_top_right_spines(ax_ad2)

    # Add titles for the two sets
    fig1.text(0.25, 0.95, "First Set of Images", ha="center", fontsize=12)
    fig1.text(0.75, 0.95, "Second Set of Images", ha="center", fontsize=12)

    plt.tight_layout()

    # Create second figure for STAs
    fig2 = plt.figure(figsize=(20, 4))  # Increased from (15, 3)
    gs_sta = fig2.add_gridspec(1, n_top_neurons)

    # Plot all STAs in a grid
    for i in range(n_top_neurons):
        ax_sta = fig2.add_subplot(gs_sta[0, i])
        sta = sorted_stas[i]
        # Ensure STA is 2D for plotting
        if len(sta.shape) == 3:
            img_sta = sta[0]  # Take first channel if 3D
        else:
            img_sta = sta
        # Convert from [-1, 1] to [0, 1] for display
        img_sta = (img_sta + 1) / 2
        ax_sta.imshow(img_sta, cmap="gray")
        for spine in ax_sta.spines.values():
            spine.set_edgecolor(colors[i])
            spine.set_linewidth(3)
        ax_sta.set_title(f"N{neuron_sort_idx[i]}", color=colors[i], fontsize=8)
        ax_sta.axis("off")

    plt.tight_layout()

    return fig1, fig2


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

    firing_rates, dot_products, adaptation_states = (
        simulator.simulate_neural_responses_vectorized(
            batch_size=args.batch_size
        )
    )

    print("Plotting example results...")
    fig1, fig2 = plot_sta_and_spikes(
        images=images,
        responses=firing_rates,
        dot_products=dot_products,
        adaptation_states=adaptation_states,
        stas=simulator.selected_stas,
        coords=simulator.rf_coords,
    )

    # Create output directory
    os.makedirs("output", exist_ok=True)

    print("Saving dataset...")
    sta_avg_correlations = calculate_sta_vs_zscore_correlations(
        images, firing_rates, simulator.selected_stas, simulator.rf_coords
    )

    # Print correlation statistics
    print("\n=== STA vs Z-score Average Image Correlations ===")
    print(f"Mean correlation: {np.mean(sta_avg_correlations):.4f}")
    print(f"Std correlation: {np.std(sta_avg_correlations):.4f}")
    print(f"Min correlation: {np.min(sta_avg_correlations):.4f}")
    print(f"Max correlation: {np.max(sta_avg_correlations):.4f}")
    print(
        f"Number of neurons with correlation > 0.5: "
        f"{np.sum(sta_avg_correlations > 0.5)}/{len(sta_avg_correlations)}"
    )
    print(
        f"Number of neurons with correlation > 0.7: "
        f"{np.sum(sta_avg_correlations > 0.7)}/{len(sta_avg_correlations)}"
    )

    save_output(
        images,
        firing_rates,
        simulator.selected_stas,
        simulator.rf_coords,
        adaptation_states,
        labels,
        args.dataset_type,
        args.sta_type,
        args.n_neurons,
        args.n_images,
        sta_avg_correlations,
    )
    print("Done.")


if __name__ == "__main__":
    main()
