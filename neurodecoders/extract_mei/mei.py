import argparse
import datetime
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "encoder"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "synthetic"))

from synthetic.simulate_response import SimulateResponse

from neurodecoders.encoder.models import ResNetEncoder, SimpleEncoder
from neurodecoders.synthetic.sta import STA


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

    def get_expected_firing_rate(self, image, target_neuron_idx):
        """Get expected firing rate using SimulateResponse"""
        if self.sta_patterns is None or self.rf_coords is None:
            raise ValueError(
                "Cannot compute expected firing rate: STA patterns or RF "
                "coordinates not available. Please provide a valid "
                "--data_path with synthetic data containing STA patterns "
                "and RF coordinates."
            )

        # If image is a patch (smaller than full image), create a full image
        # with patch at RF location
        if image.shape[-1] < 224:  # This is a patch
            # Create a full 224x224 image with zeros
            full_image = torch.zeros(
                1, 224, 224, device=self.device, dtype=torch.float32
            )

            # Get RF coordinates for the target neuron
            x, y = self.rf_coords[target_neuron_idx]
            patch_size = image.shape[-1]

            # Ensure coordinates are valid
            if x + patch_size > 224 or y + patch_size > 224:
                # Adjust coordinates if needed
                x = min(x, 224 - patch_size)
                y = min(y, 224 - patch_size)
                self.rf_coords[target_neuron_idx] = [x, y]

            # Place the patch at the RF location
            full_image[0, y : y + patch_size, x : x + patch_size] = (
                image.squeeze()
            )
            single_image = full_image
        else:
            # Image is already full size
            if image.dim() == 4:  # (1, 1, H, W) - remove batch dimension
                single_image = image.squeeze(0)  # Now (1, H, W)
            else:
                single_image = image  # Already (1, H, W)

        # Ensure image is float32
        single_image = single_image.float()

        # Use the existing SimulateResponse class

        # Temporarily disable tqdm output to suppress progress bars
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

        try:
            simulator = SimulateResponse(
                self.device,
                [single_image],
                self.sta_patterns,
                len(self.sta_patterns),
            )
            simulator.rf_coords = self.rf_coords  # Set the RF coordinates

            # Simulate response for this single image
            firing_rates, _, _ = simulator.simulate_neural_responses()
        finally:
            # Restore stdout
            sys.stdout.close()
            sys.stdout = original_stdout

        # Get the firing rate for the target neuron
        expected_firing_rate = torch.tensor(
            firing_rates[0, target_neuron_idx],
            device=self.device,
            dtype=torch.float32,
        )

        return expected_firing_rate

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

    def verify_bounds(self, image):
        """Verify and ensure image is within [-1, 1] bounds using
        normalization"""
        with torch.no_grad():
            min_val = image.min().item()
            max_val = image.max().item()

            if min_val < -1 or max_val > 1:
                image = self.normalize_to_bounds(image, -1, 1)

            return image

    def initialize_random_image(self, sta_size=None, noise_std=0.1):
        """Initialize a random noise image for optimization"""
        # Use STA pattern size if provided, otherwise default to 11x11
        if sta_size is None:
            sta_size = 11

        # Create random noise patch
        random_patch = (
            torch.randn(
                1,
                1,
                sta_size,
                sta_size,
                device=self.device,
                dtype=torch.float32,
            )
            * noise_std
        )

        # Normalize to [-1, 1] range to preserve relative relationships
        random_patch = self.normalize_to_bounds(random_patch, -1, 1)

        # Make the patch trainable
        random_patch.requires_grad_(True)
        return random_patch

    def optimize_image(
        self,
        initial_image,
        learning_rate=0.01,
        num_iterations=1000,
        regularization_weight=0.001,
        update_interval=50,
        callback=None,
        patience=100,
        factor=0.5,
        min_lr=1e-6,
    ):
        """
        Optimize the image to maximize firing rate of target neuron

        Args:
            initial_image: Starting image tensor (1, 1, H, W)
            learning_rate: Initial learning rate for optimization
            num_iterations: Number of optimization steps
            regularization_weight: Weight for L2 regularization on image
            update_interval: How often to yield intermediate results
            callback: Optional callback function for real-time updates
            patience: Number of iterations to wait before reducing LR on
                plateau
            factor: Factor by which to reduce learning rate
            min_lr: Minimum learning rate threshold

        Returns:
            optimized_image: The optimized image
            loss_history: List of loss values during optimization
            firing_rate_history: List of firing rates during optimization
        """

        # Clone the initial image and make it trainable
        image = initial_image.clone().detach().requires_grad_(True)

        # Optimizer for the image only
        optimizer = torch.optim.Adam([image], lr=learning_rate)

        # Learning rate scheduler for plateau detection
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

        # Loss history tracking
        loss_history = []
        firing_rate_history = []
        expected_firing_rate_history = []
        lr_history = []

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Get expected firing rate from simulation
            expected_firing_rate = self.get_expected_firing_rate(
                image, self.target_neuron_idx
            )

            # Get predicted firing rate from encoder
            firing_rates = self.encoder_model(image)
            predicted_firing_rate = firing_rates[0, self.target_neuron_idx]

            # Loss: minimize the difference between predicted and expected
            # firing rates
            # We want the encoder to predict high firing rates when the
            # simulation gives high firing rates
            rate_loss = F.mse_loss(predicted_firing_rate, expected_firing_rate)

            # L2 regularization on image to prevent extreme values
            l2_reg = regularization_weight * torch.norm(image)

            # Total loss
            total_loss = rate_loss + l2_reg

            # Backward pass
            total_loss.backward()

            optimizer.step()

            # Use normalization to ensure [-1, 1] range while preserving
            # relationships
            with torch.no_grad():
                # Check if bounds are exceeded
                min_val = image.min()
                max_val = image.max()

                if min_val < -1 or max_val > 1:
                    # Normalize to preserve relative relationships
                    image.data = self.normalize_to_bounds(image.data, -1, 1)

            # Record history
            loss_history.append(total_loss.item())
            firing_rate_history.append(predicted_firing_rate.item())
            expected_firing_rate_history.append(expected_firing_rate.item())
            lr_history.append(optimizer.param_groups[0]["lr"])

            # Update learning rate scheduler
            scheduler.step(total_loss)

            # Callback for real-time updates
            if callback and iteration % update_interval == 0:
                # Verify bounds before passing to callback
                bounded_image = self.verify_bounds(image.detach().clone())
                callback(
                    iteration,
                    bounded_image,
                    total_loss.item(),
                    predicted_firing_rate.item(),
                    expected_firing_rate.item(),
                    optimizer.param_groups[0]["lr"],
                )

        return (
            image.detach(),
            loss_history,
            firing_rate_history,
            expected_firing_rate_history,
            lr_history,
        )

    def compare_with_sta(self, optimized_image, sta_pattern, neuron_idx):
        """
        Compare the optimized MEI with the STA pattern for the same neuron

        Args:
            optimized_image: The optimized MEI (1, 1, H, W)
            sta_pattern: The STA pattern for the neuron (H, W)
            neuron_idx: Index of the neuron

        Returns:
            comparison_dict: Dictionary with comparison metrics
        """
        mei_np = optimized_image.squeeze().cpu().numpy()
        sta_np = sta_pattern
        # Resize STA to match MEI shape if needed
        if mei_np.shape != sta_np.shape:
            sta_np = resize(
                sta_np,
                mei_np.shape,
                order=1,
                mode="reflect",
                anti_aliasing=True,
            )

        # Calculate correlation
        correlation = np.corrcoef(mei_np.flatten(), sta_np.flatten())[0, 1]

        # Calculate cosine similarity
        cos_sim = np.dot(mei_np.flatten(), sta_np.flatten()) / (
            np.linalg.norm(mei_np.flatten()) * np.linalg.norm(sta_np.flatten())
        )

        # Calculate MSE
        mse = np.mean((mei_np - sta_np) ** 2)

        # Calculate structural similarity (SSIM-like)
        ssim_score = ssim(
            mei_np, sta_np, data_range=2.0
        )  # data_range = max - min = 1 - (-1) = 2

        return {
            "correlation": correlation,
            "cosine_similarity": cos_sim,
            "mse": mse,
            "ssim": ssim_score,
            "neuron_idx": neuron_idx,
        }


def load_encoder_model(model_path, device):
    """Load a trained encoder model"""
    try:
        # Load the saved state dict
        state_dict = torch.load(model_path, map_location=device)

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

        # Extract STA information from filename
        filename = os.path.basename(data_path)
        # Look for '_sta-' and '_n_neurons' to extract the full sta_type string
        if "_sta-" in filename and "_n_neurons" in filename:
            sta_type = filename.split("_sta-")[1].split("_n_neurons")[0]
            return images, firing_rates, sta_type
        else:
            return images, firing_rates, None
    except Exception as e:
        raise ValueError(f"Error loading data from {data_path}: {str(e)}")


def generate_sta_patterns(sta_type, n_neurons):
    """Generate STA patterns for comparison"""
    sta_generator = STA()
    sta_patterns = sta_generator.get_simulated_sta(sta_type, n_neurons)
    return sta_patterns


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
):
    """Create visualization plots for MEI optimization"""
    mei_np = optimized_image.squeeze().cpu().numpy()
    sta_np = sta_pattern
    # Resize STA to match MEI shape if needed
    if mei_np.shape != sta_np.shape:
        sta_np = resize(
            sta_np, mei_np.shape, order=1, mode="reflect", anti_aliasing=True
        )
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    im1 = axes[0, 0].imshow(mei_np, cmap="gray", vmin=-1, vmax=1)
    axes[0, 0].set_title(f"Optimized MEI (Neuron {neuron_idx})")
    axes[0, 0].axis("off")
    plt.colorbar(im1, ax=axes[0, 0])
    im2 = axes[0, 1].imshow(sta_np, cmap="gray", vmin=-1, vmax=1)
    axes[0, 1].set_title(f"STA Pattern (Neuron {neuron_idx})")
    axes[0, 1].axis("off")
    plt.colorbar(im2, ax=axes[0, 1])
    diff = mei_np - sta_np
    im3 = axes[0, 2].imshow(diff, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 2].set_title("MEI - STA Difference")
    axes[0, 2].axis("off")
    plt.colorbar(im3, ax=axes[0, 2])
    axes[1, 0].plot(loss_history)
    axes[1, 0].set_title("Optimization Loss")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(firing_rate_history)
    axes[1, 1].set_title("Firing Rate Evolution")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Firing Rate (Hz)")
    axes[1, 1].grid(True, alpha=0.3)
    metrics_names = ["Correlation", "Cosine Sim", "SSIM"]
    metrics_values = [
        comparison_metrics["correlation"],
        comparison_metrics["cosine_similarity"],
        comparison_metrics["ssim"],
    ]
    bars = axes[1, 2].bar(
        metrics_names,
        metrics_values,
        color=["skyblue", "lightgreen", "lightcoral"],
    )
    axes[1, 2].set_title("Similarity Metrics")
    axes[1, 2].set_ylabel("Score")
    axes[1, 2].set_ylim(-1, 1)
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        axes[1, 2].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
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
        default="workspace/mei_results",
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
        default=1000,
        help="Number of optimization iterations (default: 1000)",
    )
    parser.add_argument(
        "--regularization_weight",
        type=float,
        default=0.001,
        help="L2 regularization weight (default: 0.001)",
    )
    parser.add_argument(
        "--sta_size",
        type=int,
        default=11,
        help="Size of STA pattern for initialization (default: 11)",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.1,
        help="Standard deviation for random initialization (default: 0.1)",
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

    if args.data_path:
        try:
            images, firing_rates, sta_type = load_synthetic_data(
                args.data_path
            )
            sta_patterns = generate_sta_patterns(
                sta_type, firing_rates.shape[1]
            )
            # Generate RF coordinates (simple grid for now)
            n_neurons = firing_rates.shape[1]
            rf_coords = []
            for i in range(n_neurons):
                x = (i % 10) * 20 + 10  # Simple grid layout
                y = (i // 10) * 20 + 10
                rf_coords.append([x, y])
            print(f"Loaded STA patterns for {len(sta_patterns)} neurons")
            print(f"STA pattern shape: {sta_patterns[0].shape}")
            print(
                "STA pattern range: "
                f"[{sta_patterns[0].min():.3f}, {sta_patterns[0].max():.3f}]"
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

    # Initialize random image
    initial_image = optimizer.initialize_random_image(
        sta_size=args.sta_size, noise_std=args.noise_std
    )

    print(f"Starting MEI optimization for neuron {args.neuron_idx}")
    print(f"Initial image shape: {initial_image.shape}")

    # Define callback for progress updates
    def progress_callback(
        iteration, image, loss, predicted_rate, expected_rate, lr
    ):
        if iteration % 100 == 0:
            print(
                f"Iteration {iteration}: Loss={loss:.4f}, "
                f"Predicted Rate={predicted_rate:.2f}, "
                f"Expected Rate={expected_rate:.2f}, "
                f"LR={lr:.6f}"
            )

    # Run optimization
    (
        optimized_image,
        loss_history,
        firing_rate_history,
        expected_firing_rate_history,
        lr_history,
    ) = optimizer.optimize_image(
        initial_image=initial_image,
        learning_rate=args.learning_rate,
        num_iterations=args.num_iterations,
        regularization_weight=args.regularization_weight,
        callback=progress_callback,
    )

    print("Optimization completed!")
    print(f"Final loss: {loss_history[-1]:.4f}")
    print(f"Final predicted firing rate: {firing_rate_history[-1]:.2f}")

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

    # Create and save plots if requested
    if args.save_plots:
        if sta_patterns is not None and args.neuron_idx < len(sta_patterns):
            sta_pattern = sta_patterns[args.neuron_idx]
            print(f"Using STA pattern {args.neuron_idx} from loaded patterns")
            fig = plot_mei_optimization(
                optimized_image=optimized_image,
                sta_pattern=sta_pattern,
                loss_history=loss_history,
                firing_rate_history=firing_rate_history,
                comparison_metrics=comparison_metrics,
                neuron_idx=args.neuron_idx,
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
        plots_dir = "workspace/plots"
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, plot_filename)
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plots: {plot_path}")


if __name__ == "__main__":
    main()
