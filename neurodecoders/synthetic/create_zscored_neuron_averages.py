import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr


def load_synthetic_data(data_path):
    """
    Load synthetic neural response data from .npz file.
    """
    data = np.load(data_path)
    result = {
        "images": torch.from_numpy(data["images"]),
        "responses": data["responses"],
        "stas": data["stas"],
        "rf_coords": data["rf_coords"],
        "adaptation_states": data["adaptation_states"],
        "labels": torch.from_numpy(data["labels"]),
    }

    # Add correlations if they exist in the file
    if "sta_avg_correlations" in data:
        result["sta_avg_correlations"] = data["sta_avg_correlations"]

    return result


def create_zscore_neuron_average_images(images, responses, n_neurons=None):
    """
    Create average images for each neuron using z-score normalized firing
    rates as weights.

    Args:
        images (torch.Tensor): Image tensor of shape (n_images, channels,
            height, width)
        responses (np.ndarray): Firing rates of shape (n_images, n_neurons)
        n_neurons (int, optional): Number of neurons to process. If None,
            process all.

    Returns:
        np.ndarray: Average images of shape (n_neurons, channels, height,
            width)
    """
    if n_neurons is None:
        n_neurons = responses.shape[1]

    # Convert images to numpy if needed
    if isinstance(images, torch.Tensor):
        images_np = images.cpu().numpy()
    else:
        images_np = images

    # Initialize output array
    n_channels, height, width = images_np.shape[1:]
    average_images = np.zeros((n_neurons, n_channels, height, width))

    print(
        f"Creating z-score normalized average images for "
        f"{n_neurons} neurons..."
    )

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

        # Compute weighted average
        # Reshape weights for broadcasting: (n_images, 1, 1, 1)
        weights_reshaped = weights.reshape(-1, 1, 1, 1)

        # Weighted sum: (n_images, channels, height, width) *
        # (n_images, 1, 1, 1)
        weighted_images = images_np * weights_reshaped

        # Sum over images
        average_images[neuron_idx] = np.sum(weighted_images, axis=0)

        if (neuron_idx + 1) % 100 == 0:
            print(f"Processed {neuron_idx + 1}/{n_neurons} neurons")

    return average_images


def analyze_correlation_with_std(average_images, responses):
    """
    Analyze correlation between max firing rate and std of average image.

    Args:
        average_images (np.ndarray): Average images of shape (n_neurons,
            channels, height, width)
        responses (np.ndarray): Firing rates of shape (n_images, n_neurons)

    Returns:
        dict: Analysis results
    """
    # Calculate max firing rates for each neuron
    max_firing_rates = np.max(responses, axis=0)

    # Calculate std of average images for each neuron
    avg_image_stds = np.std(average_images, axis=(1, 2, 3))

    # Calculate correlation
    correlation, p_value = pearsonr(max_firing_rates, avg_image_stds)

    # Additional statistics
    mean_firing_rates = np.mean(responses, axis=0)
    std_firing_rates = np.std(responses, axis=0)

    # Correlations with other metrics
    corr_mean_fr, p_mean_fr = pearsonr(mean_firing_rates, avg_image_stds)
    corr_std_fr, p_std_fr = pearsonr(std_firing_rates, avg_image_stds)

    return {
        "max_fr_vs_avg_std_corr": correlation,
        "max_fr_vs_avg_std_p": p_value,
        "mean_fr_vs_avg_std_corr": corr_mean_fr,
        "mean_fr_vs_avg_std_p": p_mean_fr,
        "std_fr_vs_avg_std_corr": corr_std_fr,
        "std_fr_vs_avg_std_p": p_std_fr,
        "max_firing_rates": max_firing_rates,
        "avg_image_stds": avg_image_stds,
        "mean_firing_rates": mean_firing_rates,
        "std_firing_rates": std_firing_rates,
    }


def plot_sta_vs_zscore_analysis(analysis_results, responses, save_path=None):
    """
    Plot correlation analysis between STAs and z-score average images.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    sta_avg_correlations = analysis_results["sta_avg_correlations"]
    mean_firing_rates = np.mean(responses, axis=0)

    # Scatter plot of STA vs Average image correlations vs Mean firing rate
    ax.scatter(
        mean_firing_rates, sta_avg_correlations, alpha=0.6, color="blue"
    )
    ax.set_xlabel("Mean Firing Rate (Hz)")
    ax.set_ylabel("STA vs Z-score Avg Image Correlation")
    ax.set_title(
        f"STA vs Z-score Avg Image Correlations vs Mean Firing Rate\n"
        f"Mean Correlation: {analysis_results['mean_sta_avg_corr']:.4f}"
    )
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"STA vs Z-score analysis saved to {save_path}")

    return fig


def plot_neuron_average_images(
    average_images_zscore,
    responses,
    rf_coords,
    stas,
    n_plot_neurons=20,
    save_path=None,
):
    """
    Plot z-score average images and STAs for selected neurons.

    Args:
        average_images_zscore (np.ndarray): Z-score normalized average
            images of shape (n_neurons, channels, height, width)
        responses (np.ndarray): Firing rates of shape (n_images, n_neurons)
        rf_coords (np.ndarray): Receptive field coordinates of shape
            (n_neurons, 2)
        stas (np.ndarray): STAs of shape (n_neurons, channels, height, width)
        n_plot_neurons (int): Number of neurons to plot
        save_path (str, optional): Path to save the figure
    """
    # Infer RF size from STA data
    if len(stas.shape) == 3:
        rf_size = stas.shape[1]  # (n_neurons, height, width)
    else:
        rf_size = stas.shape[2]  # (n_neurons, channels, height, width)

    print(f"Inferred RF size from STA data: {rf_size}x{rf_size}")

    # Sort neurons by their maximum firing rate
    max_responses = np.max(responses, axis=0)
    neuron_sort_idx = np.argsort(max_responses)[::-1]  # Descending order
    top_neurons = neuron_sort_idx[:n_plot_neurons]

    # Create figure with 2 columns: Z-score Avg, STA
    fig, axes = plt.subplots(
        n_plot_neurons, 2, figsize=(10, 5 * n_plot_neurons)
    )
    if n_plot_neurons == 1:
        axes = axes.reshape(1, -1)

    colors = plt.cm.tab10(np.linspace(0, 1, n_plot_neurons))

    for i, neuron_idx in enumerate(top_neurons):
        # Plot z-score normalized average image
        ax1 = axes[i, 0]
        avg_img_zscore = average_images_zscore[neuron_idx]
        if avg_img_zscore.shape[0] == 1:  # Grayscale
            img_display = avg_img_zscore[0]
        else:  # RGB, take first channel for display
            img_display = avg_img_zscore[0]

        # Normalize for display
        img_display = (img_display - img_display.min()) / (
            img_display.max() - img_display.min() + 1e-8
        )
        ax1.imshow(img_display, cmap="gray")
        ax1.set_title(
            f"Neuron {neuron_idx}\n"
            f"Z-score Avg (Max FR: {max_responses[neuron_idx]:.2f})"
        )
        ax1.axis("off")

        # Add receptive field rectangle
        x, y = rf_coords[neuron_idx]
        rect = Rectangle(
            (x, y),
            rf_size,
            rf_size,
            linewidth=2,
            edgecolor=colors[i],
            facecolor="none",
            alpha=0.8,
        )
        ax1.add_patch(rect)

        # Plot STA
        ax2 = axes[i, 1]
        sta = stas[neuron_idx]
        if len(sta.shape) == 3:
            sta_display = sta[0]  # Take first channel if 3D
        else:
            sta_display = sta

        # Normalize STA for display
        sta_display = (sta_display - sta_display.min()) / (
            sta_display.max() - sta_display.min() + 1e-8
        )
        ax2.imshow(sta_display, cmap="gray")
        ax2.set_title("STA")
        ax2.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig


def main():
    """
    Main function to create top-10 average images and analyze correlations.
    """
    # Find the most recent synthetic data file
    data_dir = "workspace/datasets/synthetic"
    if not os.path.exists(data_dir):
        print(
            f"Data directory {data_dir} not found. "
            "Please run create_simulated_neural_responses.py first."
        )
        return

    # Find the most recent .npz file
    npz_files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
    if not npz_files:
        print(
            f"No .npz files found in {data_dir}. "
            "Please run create_simulated_neural_responses.py first."
        )
        return

    # Sort by modification time and take the most recent
    data_file = sorted(
        npz_files, key=lambda x: os.path.getmtime(os.path.join(data_dir, x))
    )[-1]
    data_path = os.path.join(data_dir, data_file)

    print(f"Loading data from {data_path}")
    data = load_synthetic_data(data_path)

    # Create output directory
    output_dir = "workspace/plots/neuron_averages_top10"
    os.makedirs(output_dir, exist_ok=True)

    # Create z-score normalized average images
    print("Creating z-score normalized average images for all neurons...")
    average_images_zscore = create_zscore_neuron_average_images(
        data["images"], data["responses"], n_neurons=None
    )

    # Analyze correlations
    print(
        "Analyzing correlations between firing rates and average image "
        "statistics..."
    )
    analysis_results_all = analyze_correlation_with_std(
        average_images_zscore, data["responses"]
    )

    # Analyze correlations between STAs and z-score average images
    print("Analyzing correlations between STAs and z-score average images...")

    # Check if correlations are already calculated and saved
    if "sta_avg_correlations" in data:
        print("Using pre-calculated correlations from saved data...")
        sta_avg_correlations = data["sta_avg_correlations"]
        analysis_results_sta_vs_zscore = {
            "sta_avg_correlations": sta_avg_correlations,
            "mean_sta_avg_corr": np.mean(sta_avg_correlations),
            "std_sta_avg_corr": np.std(sta_avg_correlations),
        }

        # Calculate additional statistics
        if len(data["stas"].shape) == 3:
            sta_std = np.std(data["stas"], axis=(1, 2))
        else:
            sta_std = np.std(data["stas"], axis=(1, 2, 3))

        if len(average_images_zscore.shape) == 3:
            avg_img_std = np.std(average_images_zscore, axis=(1, 2))
        else:
            avg_img_std = np.std(average_images_zscore, axis=(1, 2, 3))

        sta_std_vs_avg_std_corr, sta_std_vs_avg_std_p = pearsonr(
            sta_std, avg_img_std
        )

        analysis_results_sta_vs_zscore.update(
            {
                "sta_std_vs_avg_std_corr": sta_std_vs_avg_std_corr,
                "sta_std_vs_avg_std_p": sta_std_vs_avg_std_p,
                "sta_std": sta_std,
                "avg_img_std": avg_img_std,
            }
        )
    else:
        raise ValueError(
            "STA correlations not found in saved data. "
            "Please regenerate the synthetic data with correlations included."
        )

    # Print correlation results
    print("\n=== CORRELATION ANALYSIS ===")
    print("Z-score normalized average images:")
    print(
        f"  Max firing rate vs Average image std: "
        f"r={analysis_results_all['max_fr_vs_avg_std_corr']:.4f}, "
        f"p={analysis_results_all['max_fr_vs_avg_std_p']:.2e}"
    )
    print(
        f"  Mean firing rate vs Average image std: "
        f"r={analysis_results_all['mean_fr_vs_avg_std_corr']:.4f}, "
        f"p={analysis_results_all['mean_fr_vs_avg_std_p']:.2e}"
    )
    print(
        f"  Std firing rate vs Average image std: "
        f"r={analysis_results_all['std_fr_vs_avg_std_corr']:.4f}, "
        f"p={analysis_results_all['std_fr_vs_avg_std_p']:.2e}"
    )

    print("\nSTA vs Z-score average images:")
    print(
        f"  Mean STA vs Z-score Avg Image Correlation: "
        f"{analysis_results_sta_vs_zscore['mean_sta_avg_corr']:.4f}"
    )
    print(
        f"  STA Std vs Z-score Avg Image Std Correlation: "
        f"r={analysis_results_sta_vs_zscore['sta_std_vs_avg_std_corr']:.3f}, "
        f"p={analysis_results_sta_vs_zscore['sta_std_vs_avg_std_p']:.3e}"
    )

    # Create plots
    print("Creating visualization plots...")

    plot_neuron_average_images(
        average_images_zscore,
        data["responses"],
        data["rf_coords"],
        data["stas"],
        n_plot_neurons=20,
        save_path=f"{output_dir}/neuron_average_images_zscore.png",
    )

    plot_sta_vs_zscore_analysis(
        analysis_results_sta_vs_zscore,
        data["responses"],
        save_path=f"{output_dir}/sta_vs_zscore_analysis.png",
    )

    # Save average images
    np.save(
        f"{output_dir}/neuron_average_images_zscore.npy", average_images_zscore
    )

    # Save correlations if available
    if "sta_avg_correlations" in data:
        np.save(
            f"{output_dir}/sta_avg_correlations.npy",
            data["sta_avg_correlations"],
        )
        print(
            "Saved pre-calculated correlations to "
            f"{output_dir}/sta_avg_correlations.npy"
        )

    print(f"Results saved to {output_dir}/")

    print("Done!")

    # Display some basic statistics
    print("\nDataset statistics:")
    print(f"Number of neurons: {data['responses'].shape[1]}")
    print(f"Number of images: {data['responses'].shape[0]}")
    print(
        f"Z-score normalized average image shape: "
        f"{average_images_zscore.shape}"
    )
    print(
        f"Mean firing rate across all neurons: "
        f"{np.mean(data['responses']):.2f} Hz"
    )
    print(
        f"Max firing rate across all neurons: "
        f"{np.max(data['responses']):.2f} Hz"
    )

    print("\nZ-score normalized average image statistics:")
    print(f"Mean: {np.mean(average_images_zscore):.4f}")
    print(f"Std: {np.std(average_images_zscore):.4f}")
    print(f"Min: {np.min(average_images_zscore):.4f}")
    print(f"Max: {np.max(average_images_zscore):.4f}")


if __name__ == "__main__":
    main()
