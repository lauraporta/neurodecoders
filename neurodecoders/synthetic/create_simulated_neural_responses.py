import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from image_datasets import ImageDataset
from matplotlib.patches import Rectangle
from simulate_response import SimulateResponse
from sta import STA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
):
    output_dir = "workspace/datasets/synthetic"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/synthdata_dataset-{dataset_type}_sta-{sta_type}_n_neurons-{n_neurons}_n_images-{n_images}_datetime-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez(
        filename,
        images=images.cpu().numpy(),
        responses=responses,
        stas=stas,
        rf_coords=coords,
        adaptation_states=adaptation_states,
        labels=labels.cpu().numpy(),
    )


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
            rf_size = 63
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
            f"Image {image_sort_idx[i]} (Max Response: {image_max_responses[image_sort_idx[i]]:.2f})"
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
            rf_size = 63
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
            f"Image {image_sort_idx[i + n_plot_images]} (Max Response: {image_max_responses[image_sort_idx[i + n_plot_images]]:.2f})"
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


def plot_response_heatmaps(
    responses,
    dot_products,
    adaptation_states,
    n_plot_images=10,
    n_plot_neurons=10,
):
    """
    Plot heatmaps of firing rates, dot products, and adaptation states.
    """
    # Sort images by their maximum firing rate
    image_max_responses = np.max(responses, axis=1)
    image_sort_idx = np.argsort(image_max_responses)[::-1]
    sorted_responses = responses[image_sort_idx]
    sorted_dot_products = dot_products[image_sort_idx]
    sorted_adaptation = adaptation_states[image_sort_idx]

    # Sort neurons by their response to the highest responding image
    neuron_sort_idx = np.argsort(sorted_responses[0])[::-1][:n_plot_neurons]
    sorted_responses = sorted_responses[:n_plot_images, :][:, neuron_sort_idx]
    sorted_dot_products = sorted_dot_products[:n_plot_images, :][
        :, neuron_sort_idx
    ]
    sorted_adaptation = sorted_adaptation[:n_plot_images, :][
        :, neuron_sort_idx
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))  # Increased from (15, 5)
    im1 = axes[0].imshow(sorted_responses.T, aspect="auto", cmap="viridis")
    axes[0].set_title("Firing Rates (Hz)")
    axes[0].set_xlabel("Image")
    axes[0].set_ylabel("Neuron")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(
        sorted_dot_products.T, aspect="auto", cmap="viridis", vmin=0, vmax=1
    )
    axes[1].set_title("Dot Products")
    axes[1].set_xlabel("Image")
    axes[1].set_ylabel("Neuron")
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(
        sorted_adaptation.T, aspect="auto", cmap="viridis", vmin=0, vmax=1
    )
    axes[2].set_title("Adaptation States")
    axes[2].set_xlabel("Image")
    axes[2].set_ylabel("Neuron")
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    return fig


def plot_response_heatmaps_all(responses, dot_products, adaptation_states):
    """
    Plot heatmaps of firing rates, dot products, and adaptation states for all images and neurons.
    """
    fig, axes = plt.subplots(1, 3, figsize=(25, 8))  # Increased from (18, 5)
    im1 = axes[0].imshow(responses.T, aspect="auto", cmap="viridis")
    axes[0].set_title("Firing Rates (Hz)")
    axes[0].set_xlabel("Image")
    axes[0].set_ylabel("Neuron")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(
        dot_products.T, aspect="auto", cmap="viridis", vmin=0, vmax=1
    )
    axes[1].set_title("Dot Products")
    axes[1].set_xlabel("Image")
    axes[1].set_ylabel("Neuron")
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(
        adaptation_states.T, aspect="auto", cmap="viridis", vmin=0, vmax=1
    )
    axes[2].set_title("Adaptation States")
    axes[2].set_xlabel("Image")
    axes[2].set_ylabel("Neuron")
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    return fig


def plot_response_histograms(responses, dot_products, adaptation_states):
    """
    Plot histograms of firing rates, dot products, and adaptation states across all neurons and images.
    Neurons are sorted by their response to the first image.
    """
    # Sort neurons by their response to the first image
    neuron_sort_idx = np.argsort(responses[0])[::-1]  # Descending order
    sorted_responses = responses[:, neuron_sort_idx]
    sorted_dot_products = dot_products[:, neuron_sort_idx]
    sorted_adaptation = adaptation_states[:, neuron_sort_idx]

    fig, axes = plt.subplots(1, 3, figsize=(25, 8))  # Increased from (18, 5)

    # Firing rate histogram
    axes[0].hist(sorted_responses.flatten(), bins=50, color="C0", alpha=0.8)
    axes[0].set_title("Firing Rate Distribution")
    axes[0].set_xlabel("Firing Rate (Hz)")
    axes[0].set_ylabel("Count")

    # Dot product histogram
    axes[1].hist(sorted_dot_products.flatten(), bins=50, color="C1", alpha=0.8)
    axes[1].set_title("Dot Product Distribution")
    axes[1].set_xlabel("Dot Product")
    axes[1].set_ylabel("Count")

    # Adaptation state histogram
    axes[2].hist(sorted_adaptation.flatten(), bins=50, color="C2", alpha=0.8)
    axes[2].set_title("Adaptation State Distribution")
    axes[2].set_xlabel("Adaptation State")
    axes[2].set_ylabel("Count")

    plt.tight_layout()
    return fig


def plot_neural_correlations(responses):
    """
    Plot correlation matrix of neural firing rates across images.
    """
    # Compute correlation matrix between neurons
    corr_matrix = np.corrcoef(
        responses.T
    )  # Transpose to get neuron-neuron correlations

    fig, ax = plt.subplots(figsize=(15, 12))  # Increased from (10, 8)
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("Neural Firing Rate Correlations")
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Neuron")
    plt.colorbar(im, ax=ax, label="Pearson Correlation")

    plt.tight_layout()
    return fig


def main():
    n_images = 1000
    n_neurons = 1000

    print("Loading data and model...")
    images, labels = ImageDataset().get_data("cifar10", n_images=n_images)
    stas = STA().get_simulated_sta("perlin_noise_patterns,11,11")

    print("Generating responses...")
    simulator = SimulateResponse(device, images, stas, n_neurons)
    firing_rates, dot_products, adaptation_states = (
        simulator.simulate_neural_responses()
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

    # Add heatmap plot (all neurons, all images)
    fig3 = plot_response_heatmaps_all(
        firing_rates, dot_products, adaptation_states
    )
    plots_dir = "workspace/plots/analysis"
    os.makedirs(plots_dir, exist_ok=True)
    fig3.savefig(f"{plots_dir}/heatmaps_all.png")

    # Add histogram plot
    fig4 = plot_response_histograms(
        firing_rates, dot_products, adaptation_states
    )
    fig4.savefig(f"{plots_dir}/response_histograms.png")

    # Add neural correlation plot
    fig5 = plot_neural_correlations(firing_rates)
    fig5.savefig(f"{plots_dir}/neural_correlations.png")

    print("Saving dataset...")
    os.makedirs("data", exist_ok=True)
    save_output(
        images,
        firing_rates,
        simulator.selected_stas,
        simulator.rf_coords,
        adaptation_states,
        labels,
        "cifar10",
        "perlin_noise_patterns,11,11",
        n_neurons,
        n_images,
    )
    print("Done.")


if __name__ == "__main__":
    main()
