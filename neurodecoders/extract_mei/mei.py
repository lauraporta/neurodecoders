import argparse
import datetime
import os
import sys

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "encoder"))

from neurodecoders.encoder.models import ResNetEncoder, SimpleEncoder
from neurodecoders.paths import get_path


class MEIOptimizer:
    """
    Maximal Exciting Image (MEI) optimizer.
    Optimizes a random noise image to maximally excite a specific neuron.
    """

    def __init__(
        self,
        device,
        encoder_model,
        target_neuron_idx=0,
        sta_patterns=None,
        rf_coords=None,
    ):
        self.device = device
        self.encoder_model = encoder_model
        self.target_neuron_idx = target_neuron_idx
        self.sta_patterns = sta_patterns
        self.rf_coords = rf_coords

        # Set encoder to evaluation mode and freeze all parameters
        self.encoder_model.eval()
        for param in self.encoder_model.parameters():
            param.requires_grad = False

    def normalize_to_bounds(self, image, min_val=-1, max_val=1):
        """Normalize image to [min_val, max_val] range while preserving
        relative relationships"""
        with torch.no_grad():
            current_min = image.min()
            current_max = image.max()

            # Avoid division by zero
            if current_max == current_min:
                return image * 0  # Return zeros if all values are the same

            # Normalize to [0, 1] first, then scale to [min_val, max_val]
            normalized = (image - current_min) / (current_max - current_min)
            scaled = normalized * (max_val - min_val) + min_val

            return scaled

    def apply_tough_avg_pooling(self, image, pool_size=8):
        """Apply tough average pooling to prevent pixel-level artifacts."""
        # Apply average pooling with reflection padding to maintain size
        # Use reflection padding to avoid edge artifacts
        pad_size = pool_size // 2
        padded_image = F.pad(
            image, (pad_size, pad_size, pad_size, pad_size), mode="reflect"
        )

        # Apply average pooling
        pooled = F.avg_pool2d(padded_image, kernel_size=pool_size, stride=1)

        # Upsample back to original size using nearest neighbor interpolation
        upsampled = F.interpolate(pooled, size=image.shape[2:], mode="nearest")

        return upsampled

    def initialize_random_image(self, noise_std=0.1):
        """Initialize a random noise image for optimization"""

        # Create random noise image of full CIFAR-10 size (224x224)
        random_image = (
            torch.randn(
                1,
                1,
                224,
                224,
                device=self.device,
                dtype=torch.float32,
            )
            * noise_std
        )

        # Normalize to [-1, 1] range to preserve relative relationships
        random_image = self.normalize_to_bounds(random_image, -1, 1)

        # Make the image trainable
        random_image.requires_grad_(True)
        return random_image

    def optimize_image(
        self,
        initial_image,
        learning_rate=0.01,
        num_iterations=1000,
        update_interval=50,
        callback=None,
        store_images=False,
        pool_size=8,
    ):
        """
        Optimize the image to maximize firing rate of target neuron

        Args:
            initial_image: Starting image tensor (1, 1, H, W)
            learning_rate: Learning rate for optimization
            num_iterations: Number of optimization steps
            update_interval: How often to yield intermediate results
            callback: Optional callback function for real-time updates
            store_images: Whether to store images at every step
            for GIF creation

        Returns:
            optimized_image: The optimized image
            loss_history: List of loss values during optimization
            firing_rate_history: List of firing rates during optimization
            image_history: List of images at each step (if store_images=True)
        """

        # Clone the initial image and make it trainable
        image = initial_image.clone().detach().requires_grad_(True)

        # Optimizer for the image only
        optimizer = torch.optim.Adam([image], lr=learning_rate)

        # Loss history tracking
        loss_history = []
        firing_rate_history = []
        image_history = [] if store_images else None

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Apply tough average pooling regularization
            # to prevent pixel-level artifacts
            pooled_image = self.apply_tough_avg_pooling(
                image, pool_size=pool_size
            )
            image_for_encoder = 0.5 * image + 0.5 * pooled_image

            # Get predicted firing rate from encoder
            firing_rates = self.encoder_model(image_for_encoder)
            predicted_firing_rate = firing_rates[0, self.target_neuron_idx]

            # Loss: negative firing rate (we want to maximize firing rate)
            loss = -predicted_firing_rate

            # Backward pass
            loss.backward()

            optimizer.step()

            # Ensure image stays within [-1, 1] bounds
            with torch.no_grad():
                min_val = image.min()
                max_val = image.max()

                if min_val < -1 or max_val > 1:
                    # Normalize to preserve relative relationships
                    image.data = self.normalize_to_bounds(image.data, -1, 1)

            # Record history
            loss_history.append(loss.item())
            firing_rate_history.append(predicted_firing_rate.item())

            # Store image for GIF creation
            if store_images:
                image_history.append(image.detach().cpu().numpy())

            # Callback for real-time updates
            if callback and iteration % update_interval == 0:
                callback(
                    iteration,
                    image.detach().clone(),
                    loss.item(),
                    predicted_firing_rate.item(),
                )

        return (
            image.detach(),
            loss_history,
            firing_rate_history,
            image_history,
        )

    def compare_with_sta(self, optimized_image, sta_pattern, neuron_idx):
        """
        Compare the optimized MEI with the STA pattern for the same neuron

        Args:
            optimized_image: The optimized MEI (1, 1, 224, 224)
            sta_pattern: The STA pattern for the neuron (H, W)
            neuron_idx: Index of the neuron

        Returns:
            comparison_dict: Dictionary with comparison metrics
        """
        # Extract the patch from the full optimized image
        # based on RF coordinates
        mei_np = optimized_image.squeeze().cpu().numpy()  # (224, 224)
        sta_np = sta_pattern

        # Get patch size from STA pattern
        patch_size = sta_np.shape[0]  # Assuming square patch

        # Extract patch from MEI based on RF coordinates
        if self.rf_coords is not None and neuron_idx < len(self.rf_coords):
            x, y = self.rf_coords[neuron_idx]
            # Ensure coordinates are valid
            x = min(max(0, x), 224 - patch_size)
            y = min(max(0, y), 224 - patch_size)
            mei_patch = mei_np[y : y + patch_size, x : x + patch_size]
        else:
            # Center the patch if no RF coordinates
            start_y = (224 - patch_size) // 2
            start_x = (224 - patch_size) // 2
            mei_patch = mei_np[
                start_y : start_y + patch_size, start_x : start_x + patch_size
            ]

        # Resize STA to match MEI patch shape if needed
        if mei_patch.shape != sta_np.shape:
            sta_np = resize(
                sta_np,
                mei_patch.shape,
                order=1,
                mode="reflect",
                anti_aliasing=True,
            )

        # Calculate correlation
        correlation = np.corrcoef(mei_patch.flatten(), sta_np.flatten())[0, 1]

        # Calculate cosine similarity
        cos_sim = np.dot(mei_patch.flatten(), sta_np.flatten()) / (
            np.linalg.norm(mei_patch.flatten())
            * np.linalg.norm(sta_np.flatten())
        )

        # Calculate MSE
        mse = np.mean((mei_patch - sta_np) ** 2)

        # Calculate structural similarity (SSIM-like)
        ssim_score = ssim(
            mei_patch, sta_np, data_range=2.0
        )  # data_range = max - min = 1 - (-1) = 2

        return {
            "correlation": correlation,
            "cosine_similarity": cos_sim,
            "mse": mse,
            "ssim": ssim_score,
            "neuron_idx": neuron_idx,
        }

    def create_optimization_gif(
        self, image_history, output_path, fps=10, subsample_factor=1
    ):
        """
        Create a GIF from the optimization history

        Args:
            image_history: List of numpy arrays (iterations, 1, 1, H, W)
            output_path: Path to save the GIF
            fps: Frames per second for the GIF
            subsample_factor: Only use every nth frame to reduce file size
        """
        if not image_history:
            print("No image history provided for GIF creation")
            return

        # Subsample frames if requested
        if subsample_factor > 1:
            image_history = image_history[::subsample_factor]

        # Convert images to uint8 format for GIF
        frames = []
        for i, img in enumerate(image_history):
            # Squeeze to remove batch and channel dimensions
            img_2d = img.squeeze()

            # Normalize to [0, 255] range
            img_norm = ((img_2d + 1) * 127.5).astype(np.uint8)

            frames.append(img_norm)

        # Create GIF
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"GIF saved to: {output_path}")
        print(f"GIF contains {len(frames)} frames at {fps} fps")


def load_encoder_model(model_path, device):
    """Load a trained encoder model from either .pth or .ckpt files"""
    try:
        # Check if it's a PyTorch Lightning checkpoint (.ckpt file)
        if model_path.endswith(".ckpt"):
            # Load PyTorch Lightning checkpoint
            checkpoint = torch.load(model_path, map_location=device)

            # Extract state dict from Lightning checkpoint
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                # If no 'state_dict' key, assume the
                # checkpoint is the state dict
                state_dict = checkpoint

            # Remove 'model.' prefix if it exists
            # (Lightning often wraps models)
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("model."):
                    new_key = key[6:]  # Remove 'model.' prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict

        else:
            # Load regular PyTorch state dict (.pth file)
            state_dict = torch.load(model_path, map_location=device)

            # Handle nested model structure in state dict
            # If keys have 'model.' prefix, remove it
            if any(k.startswith("model.") for k in state_dict.keys()):
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("model."):
                        new_key = key[6:]  # Remove 'model.' prefix
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                state_dict = new_state_dict

        # Determine model type and number of output neurons
        # Check for ResNet encoder (has firing_head layers)
        if any("firing_head" in k for k in state_dict.keys()):
            # ResNet encoder
            if "firing_head.6.weight" in state_dict:
                saved_out_neurons = state_dict["firing_head.6.weight"].shape[0]
            elif "model.firing_head.6.weight" in state_dict:
                saved_out_neurons = state_dict[
                    "model.firing_head.6.weight"
                ].shape[0]
            else:
                # Find the last firing_head layer
                firing_head_keys = [
                    k
                    for k in state_dict.keys()
                    if "firing_head" in k and "weight" in k
                ]
                if firing_head_keys:
                    last_firing_head_key = sorted(firing_head_keys)[-1]
                    saved_out_neurons = state_dict[last_firing_head_key].shape[
                        0
                    ]
                else:
                    raise ValueError(
                        "Could not determine ResNet model architecture "
                        "from saved weights"
                    )

            # Create ResNet encoder
            model = ResNetEncoder(saved_out_neurons, resnet_type="resnet18")

        else:
            # Simple encoder (has fc layers)
            if "model.fc.6.weight" in state_dict:
                saved_out_neurons = state_dict["model.fc.6.weight"].shape[0]
            elif "fc.6.weight" in state_dict:
                saved_out_neurons = state_dict["fc.6.weight"].shape[0]
            elif "fc.2.weight" in state_dict:
                saved_out_neurons = state_dict["fc.2.weight"].shape[0]
            elif "model.fc.2.weight" in state_dict:
                saved_out_neurons = state_dict["model.fc.2.weight"].shape[0]
            else:
                # Try to find any fc layer weight
                fc_keys = [
                    k for k in state_dict.keys() if "fc" in k and "weight" in k
                ]
                if fc_keys:
                    # Get the last fc layer (highest index)
                    last_fc_key = sorted(fc_keys)[-1]
                    saved_out_neurons = state_dict[last_fc_key].shape[0]
                else:
                    raise ValueError(
                        "Could not determine model architecture "
                        "from saved weights"
                    )

            # Create SimpleEncoder model
            model = SimpleEncoder(saved_out_neurons)

        # Load the weights
        model.load_state_dict(state_dict)
        model.to(device)

        return model
    except Exception as e:
        raise ValueError(f"Error loading model from {model_path}: {str(e)}")


def load_synthetic_data(data_path):
    """Load synthetic data to get STA patterns"""
    try:
        data = np.load(data_path)
        images = data["images"]
        firing_rates = data["responses"]

        # Load actual STAs from the dataset
        stas = data["stas"]

        # Load actual RF coordinates from the dataset
        rf_coords = data["rf_coords"]

        # Extract STA information from filename
        filename = os.path.basename(data_path)
        # Look for '_sta-' and '_n_neurons' to extract the full sta_type string
        if "_sta-" in filename and "_n_neurons" in filename:
            sta_type = filename.split("_sta-")[1].split("_n_neurons")[0]
        else:
            sta_type = None

        # Load STA average correlations if available
        sta_avg_correlations = None
        if "sta_avg_correlations" in data:
            sta_avg_correlations = data["sta_avg_correlations"]

        return (
            images,
            firing_rates,
            stas,
            rf_coords,
            sta_type,
            sta_avg_correlations,
        )
    except Exception as e:
        raise ValueError(f"Error loading data from {data_path}: {str(e)}")


def save_mei_results(
    optimized_image,
    loss_history,
    firing_rate_history,
    comparison_metrics,
    model_path,
    neuron_idx,
    output_dir="data",
):
    """Save MEI optimization results"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the optimized image
    image_filename = (
        f"mei_image_model-"
        f"{os.path.basename(model_path).replace('.pth', '')}_neuron-"
        f"{neuron_idx}_datetime-{timestamp}.npy"
    )
    image_path = os.path.join(output_dir, image_filename)
    np.save(image_path, optimized_image.squeeze().cpu().numpy())

    # Save optimization history
    history_filename = (
        f"mei_history_model-"
        f"{os.path.basename(model_path).replace('.pth', '')}_neuron-"
        f"{neuron_idx}_datetime-{timestamp}.npz"
    )
    history_path = os.path.join(output_dir, history_filename)
    np.savez(
        history_path,
        loss_history=np.array(loss_history),
        firing_rate_history=np.array(firing_rate_history),
        comparison_metrics=comparison_metrics,
    )

    return image_path, history_path


def plot_mei_optimization(
    optimized_image,
    sta_pattern,
    loss_history,
    firing_rate_history,
    comparison_metrics,
    neuron_idx,
    sta_avg_correlation=None,
    rf_coords=None,
):
    """Create visualization plots for MEI optimization"""
    mei_np = optimized_image.squeeze().cpu().numpy()  # (224, 224)
    sta_np = sta_pattern

    # Extract the patch from the full optimized image based on RF coordinates
    patch_size = sta_np.shape[0]  # Assuming square patch

    if rf_coords is not None and neuron_idx < len(rf_coords):
        x, y = rf_coords[neuron_idx]
        # Ensure coordinates are valid
        x = min(max(0, x), 224 - patch_size)
        y = min(max(0, y), 224 - patch_size)
        mei_patch = mei_np[y : y + patch_size, x : x + patch_size]
    else:
        # Center the patch if no RF coordinates
        start_y = (224 - patch_size) // 2
        start_x = (224 - patch_size) // 2
        mei_patch = mei_np[
            start_y : start_y + patch_size, start_x : start_x + patch_size
        ]

    # Resize STA to match MEI patch shape if needed
    if mei_patch.shape != sta_np.shape:
        sta_np = resize(
            sta_np,
            mei_patch.shape,
            order=1,
            mode="reflect",
            anti_aliasing=True,
        )

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot full optimized image
    im1 = axes[0, 0].imshow(mei_np, cmap="gray", vmin=-1, vmax=1)
    axes[0, 0].set_title(f"Full Optimized MEI (Neuron {neuron_idx})")
    axes[0, 0].axis("off")
    plt.colorbar(im1, ax=axes[0, 0])

    # Add box around the patch region
    if rf_coords is not None and neuron_idx < len(rf_coords):
        x, y = rf_coords[neuron_idx]
        # Ensure coordinates are valid
        x = min(max(0, x), 224 - patch_size)
        y = min(max(0, y), 224 - patch_size)
    else:
        # Center the patch if no RF coordinates
        y = (224 - patch_size) // 2
        x = (224 - patch_size) // 2

    # Draw rectangle around the patch
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (x, y),
        patch_size,
        patch_size,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )
    axes[0, 0].add_patch(rect)

    # Plot MEI patch (extracted from RF region)
    im2 = axes[0, 1].imshow(mei_patch, cmap="gray", vmin=-1, vmax=1)
    axes[0, 1].set_title(f"MEI Patch (Neuron {neuron_idx})")
    axes[0, 1].axis("off")
    plt.colorbar(im2, ax=axes[0, 1])

    # Plot STA
    im3 = axes[0, 2].imshow(sta_np, cmap="gray", vmin=-1, vmax=1)
    axes[0, 2].set_title(f"STA Pattern (Neuron {neuron_idx})")
    axes[0, 2].axis("off")
    plt.colorbar(im3, ax=axes[0, 2])

    # Plot firing rate evolution
    axes[1, 0].plot(firing_rate_history)
    axes[1, 0].set_title("Firing Rate Evolution")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Firing Rate (Hz)")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot similarity metrics
    metrics_names = ["Correlation", "Cosine Sim", "SSIM"]
    metrics_values = [
        comparison_metrics["correlation"],
        comparison_metrics["cosine_similarity"],
        comparison_metrics["ssim"],
    ]

    # Add benchmark correlation if available
    if sta_avg_correlation is not None:
        metrics_names.append("Z-score Benchmark")
        metrics_values.append(sta_avg_correlation)
        colors = ["skyblue", "lightgreen", "lightcoral", "gold"]
    else:
        colors = ["skyblue", "lightgreen", "lightcoral"]

    bars = axes[1, 1].bar(
        metrics_names,
        metrics_values,
        color=colors,
    )
    axes[1, 1].set_title("Similarity Metrics")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_ylim(-1, 1)

    # Add benchmark line if available
    if sta_avg_correlation is not None:
        axes[1, 1].axhline(
            y=sta_avg_correlation,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Z-score Benchmark: {sta_avg_correlation:.3f}",
        )
        axes[1, 1].legend()

    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    # Plot loss evolution
    axes[1, 2].plot(loss_history)
    axes[1, 2].set_title("Loss Evolution")
    axes[1, 2].set_xlabel("Iteration")
    axes[1, 2].set_ylabel("Loss (-Firing Rate)")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Main function to run MEI optimization from command line"""
    parser = argparse.ArgumentParser(
        description="Run MEI optimization on a trained encoder model"
    )

    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained encoder model (.pth file)",
    )

    # Optional arguments
    parser.add_argument(
        "--neuron_idx",
        type=int,
        default=0,
        help="Index of the target neuron to optimize for (default: 0)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=get_path("workspace/mei_results"),
        help="Directory to save results (default: workspace/mei_results)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for optimization (default: 0.01)",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10000,
        help="Number of optimization iterations (default: 1000)",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=5,
        help="Standard deviation for random initialization (default: 0.1)",
    )
    parser.add_argument(
        "--pool_size",
        type=int,
        default=8,
        help="Average pooling kernel size for regularization (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('cpu', 'cuda', or 'auto' for automatic)",
    )
    parser.add_argument(
        "--save_plots", action="store_true", help="Save optimization plots"
    )
    parser.add_argument(
        "--create_gif",
        action="store_true",
        help="Create GIF of optimization process",
    )
    parser.add_argument(
        "--gif_fps",
        type=int,
        default=500,
        help="Frames per second for GIF (default: 10)",
    )
    parser.add_argument(
        "--gif_subsample",
        type=int,
        default=1,
        help="Subsample factor for GIF (use every nth frame, default: 1)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to synthetic data file for STA patterns (optional)",
    )

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Loading model from: {args.model_path}")

    # Load the encoder model
    try:
        encoder_model = load_encoder_model(args.model_path, device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load STA patterns if data path is provided
    sta_patterns = None
    rf_coords = None
    sta_type = None
    sta_avg_correlations = None

    if args.data_path:
        try:
            (
                images,
                firing_rates,
                stas,
                rf_coords,
                sta_type,
                sta_avg_correlations,
            ) = load_synthetic_data(args.data_path)

            # Use actual STAs from the dataset
            sta_patterns = stas

            print(
                f"Loaded actual STAs from dataset for "
                f"{len(sta_patterns)} neurons"
            )
            print(f"STA pattern shape: {sta_patterns[0].shape}")
            print(
                "STA pattern range: "
                f"[{sta_patterns[0].min():.3f}, {sta_patterns[0].max():.3f}]"
            )
            print(f"RF coordinates shape: {rf_coords.shape}")
            print(
                f"RF coordinates range: x=[{rf_coords[:, 0].min()}, "
                f"{rf_coords[:, 0].max()}], y=[{rf_coords[:, 1].min()}, "
                f"{rf_coords[:, 1].max()}]"
            )

            # Print benchmark correlation info if available
            if sta_avg_correlations is not None:
                print(
                    f"Loaded STA average correlations for "
                    f"{len(sta_avg_correlations)} neurons"
                )
                print(
                    f"Mean benchmark correlation: "
                    f"{np.mean(sta_avg_correlations):.3f}"
                )
                print(
                    f"Benchmark correlation for neuron {args.neuron_idx}: "
                    f"{sta_avg_correlations[args.neuron_idx]:.3f}"
                )
        except Exception as e:
            print(
                "Warning: Could not load STA "
                f"patterns from {args.data_path}: {e}"
            )
            print("Continuing without STA patterns...")

    # Create MEI optimizer
    optimizer = MEIOptimizer(
        device=device,
        encoder_model=encoder_model,
        target_neuron_idx=args.neuron_idx,
        sta_patterns=sta_patterns,
        rf_coords=rf_coords,
    )

    initial_image = optimizer.initialize_random_image(noise_std=args.noise_std)

    print(f"Starting MEI optimization for neuron {args.neuron_idx}")
    print(f"Initial image shape: {initial_image.shape}")

    # Define callback for progress updates
    def progress_callback(iteration, image, loss, predicted_rate):
        if iteration % 100 == 0:
            print(
                f"Iteration {iteration}: Loss={loss:.4f}, "
                f"Firing Rate={predicted_rate:.2f}"
            )

    # Run optimization
    (
        optimized_image,
        loss_history,
        firing_rate_history,
        image_history,
    ) = optimizer.optimize_image(
        initial_image=initial_image,
        learning_rate=args.learning_rate,
        num_iterations=args.num_iterations,
        callback=progress_callback,
        store_images=args.create_gif,
        pool_size=args.pool_size,
    )

    print("Optimization completed!")
    print(f"Final loss: {loss_history[-1]:.4f}")
    print(f"Final firing rate: {firing_rate_history[-1]:.2f}")

    # Compare with STA if available
    if sta_patterns is not None and args.neuron_idx < len(sta_patterns):
        sta_pattern = sta_patterns[args.neuron_idx]
        comparison_metrics = optimizer.compare_with_sta(
            optimized_image, sta_pattern, args.neuron_idx
        )
        print("Comparison with STA:")
        print(f"  Correlation: {comparison_metrics['correlation']:.3f}")
        print(
            f"  Cosine Similarity: "
            f"{comparison_metrics['cosine_similarity']:.3f}"
        )
        print(f"  SSIM: {comparison_metrics['ssim']:.3f}")

        # Compare with benchmark if available
        if sta_avg_correlations is not None and args.neuron_idx < len(
            sta_avg_correlations
        ):
            benchmark_corr = sta_avg_correlations[args.neuron_idx]
            print(f"  Z-score Benchmark Correlation: {benchmark_corr:.3f}")
            print(
                f"MEI vs Benchmark: {comparison_metrics['correlation']:.3f} "
                f"vs {benchmark_corr:.3f}"
            )
    else:
        # Raise error if STA patterns are not available
        if sta_patterns is None:
            raise ValueError(
                f"No STA patterns available. Please provide a "
                "valid --data_path with synthetic data "
                f"containing STA patterns for neuron {args.neuron_idx}."
            )
        else:
            raise ValueError(
                f"Neuron index {args.neuron_idx} out of range for STA "
                f"patterns (len={len(sta_patterns)}). Please provide data "
                f"with at least {args.neuron_idx + 1} neurons or use a "
                f"smaller neuron index."
            )

    # Save results
    image_path, history_path = save_mei_results(
        optimized_image=optimized_image,
        loss_history=loss_history,
        firing_rate_history=firing_rate_history,
        comparison_metrics=comparison_metrics,
        model_path=args.model_path,
        neuron_idx=args.neuron_idx,
        output_dir=args.output_dir,
    )

    print("Results saved:")
    print(f"  Image: {image_path}")
    print(f"  History: {history_path}")

    # Create GIF if requested
    if args.create_gif and image_history is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = (
            os.path.basename(args.model_path)
            .replace(".pth", "")
            .replace(".ckpt", "")
        )
        gif_filename = (
            f"mei_optimization_model-{model_name}_neuron-"
            f"{args.neuron_idx}_{timestamp}.gif"
        )
        gif_dir = get_path("workspace/gifs")
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, gif_filename)

        optimizer.create_optimization_gif(
            image_history=image_history,
            output_path=gif_path,
            fps=args.gif_fps,
            subsample_factor=args.gif_subsample,
        )
        print(f"  GIF: {gif_path}")

    # Create and save plots if requested
    if args.save_plots:
        if sta_patterns is not None and args.neuron_idx < len(sta_patterns):
            sta_pattern = sta_patterns[args.neuron_idx]
            print(f"Using STA pattern {args.neuron_idx} from loaded dataset")

            # Get benchmark correlation if available
            benchmark_corr = None
            if sta_avg_correlations is not None and args.neuron_idx < len(
                sta_avg_correlations
            ):
                benchmark_corr = sta_avg_correlations[args.neuron_idx]

            fig = plot_mei_optimization(
                optimized_image=optimized_image,
                sta_pattern=sta_pattern,
                loss_history=loss_history,
                firing_rate_history=firing_rate_history,
                comparison_metrics=comparison_metrics,
                neuron_idx=args.neuron_idx,
                sta_avg_correlation=benchmark_corr,
                rf_coords=rf_coords,
            )
        else:
            # Raise error if STA patterns are not available for plotting
            if sta_patterns is None:
                raise ValueError(
                    "Cannot create plots: No STA patterns available. "
                    f"Please provide a valid --data_path "
                    f"with synthetic data containing STA patterns for "
                    f"neuron {args.neuron_idx}."
                )
            else:
                raise ValueError(
                    f"Cannot create plots: Neuron index {args.neuron_idx} "
                    f"out of range for STA patterns "
                    f"(len={len(sta_patterns)}). Please provide data with "
                    f"at least {args.neuron_idx + 1} "
                    f"neurons or use a smaller neuron index."
                )

        model_name = os.path.basename(args.model_path).replace(".pth", "")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = (
            f"mei_plots_model-{model_name}_neuron-"
            f"{args.neuron_idx}_{timestamp}.png"
        )
        plots_dir = get_path("workspace/plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, plot_filename)
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plots: {plot_path}")


if __name__ == "__main__":
    main()
