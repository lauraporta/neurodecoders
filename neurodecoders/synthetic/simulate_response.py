import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class SimulateResponse:
    def __init__(self, device, images, stas, n_neurons):
        self.device = device
        self.images = images
        self.n_images = len(images)
        self.stas = stas
        self.n_neurons = n_neurons
        self.rf_size = stas.shape[1]

        self.selected_indices = np.random.choice(stas.shape[0], n_neurons)
        self.selected_stas = stas[self.selected_indices]
        self.stas_tensor = torch.tensor(
            self.selected_stas, dtype=torch.float32
        ).to(self.device)

        self.rf_coords = np.random.randint(
            0, 224 - self.rf_size, size=(n_neurons, 2)
        )

    def estimate_memory_usage(self, batch_size=100):
        """
        Estimate GPU memory usage for the simulation.

        Args:
            batch_size: Number of images to process in each batch

        Returns:
            dict: Memory usage estimates in GB
        """
        # Memory per patch: batch_size * n_neurons * 1 * rf_size * rf_size * 4
        patch_memory_gb = (
            batch_size * self.n_neurons * 1 * self.rf_size * self.rf_size * 4
        ) / (1024**3)

        # Memory for other tensors (approximate)
        other_tensors_gb = (
            batch_size * self.n_neurons * 4 * 4  # Various intermediate tensors
        ) / (1024**3)

        total_memory_gb = patch_memory_gb + other_tensors_gb

        return {
            "patch_memory_gb": patch_memory_gb,
            "other_tensors_gb": other_tensors_gb,
            "total_memory_gb": total_memory_gb,
            "suggested_batch_size": self._suggest_batch_size(),
        }

    def _suggest_batch_size(self, max_memory_gb=15):
        """
        Suggest an appropriate batch size based on available GPU memory.

        Args:
            max_memory_gb: Maximum GPU memory to use (default: 15 GB)

        Returns:
            int: Suggested batch size
        """
        # Memory per image-neuron pair: n_neurons * 1 * rf_size * rf_size * 4
        memory_per_image = (
            self.n_neurons * 1 * self.rf_size * self.rf_size * 4
        ) / (1024**3)

        # Add some overhead for other tensors
        memory_per_image *= 1.5

        suggested_batch_size = int(max_memory_gb / memory_per_image)

        # Ensure reasonable bounds
        suggested_batch_size = max(1, min(suggested_batch_size, 1000))

        return suggested_batch_size

    def simulate_neural_responses_vectorized(
        self, noise_level=0.1, batch_size=100
    ):
        """
        Memory-efficient version of neural response simulation using batch
        processing.
        """
        print("Preparing data for batch computation...")

        # Generate neuron-specific parameters with physiological constraints
        # Baseline rates: mostly very low (0.1-2 Hz), some higher
        baselines = (
            torch.exp(torch.randn(self.n_neurons, device=self.device) * 0.01)
            * 0.01
        )

        # Thresholds: log-normal distribution for more realistic,
        # skewed thresholds
        thresholds = (
            torch.exp(torch.randn(self.n_neurons, device=self.device) * 0.3)
            * 0.3
        )

        # Maximum firing rates: respecting physiological limits
        # Most neurons max out at 100-200 Hz, with some exceptions
        max_rates = (
            torch.exp(torch.randn(self.n_neurons, device=self.device) * 0.3)
            * 100
        )

        # Pre-allocate output tensors on CPU (will be moved to GPU in batches)
        firing_rates = np.zeros((self.n_images, self.n_neurons))
        dot_products = np.zeros((self.n_images, self.n_neurons))

        # Process images in batches to avoid memory overflow
        n_batches = (self.n_images + batch_size - 1) // batch_size

        print(
            f"Processing {self.n_images} images in {n_batches} batches of "
            f"{batch_size}"
        )

        for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, self.n_images)
            batch_size_actual = end_idx - start_idx

            # Pre-extract receptive field patches for this batch only
            print(
                f"  Extracting patches for batch {batch_idx + 1}/"
                f"{n_batches}..."
            )
            patches = torch.zeros(
                (
                    batch_size_actual,
                    self.n_neurons,
                    1,
                    self.rf_size,
                    self.rf_size,
                ),
                device=self.device,
            )

            for i in range(batch_size_actual):
                image_idx = start_idx + i
                image = self.images[image_idx].to(self.device)
                for n in range(self.n_neurons):
                    x, y = self.rf_coords[n]
                    patches[i, n, 0] = image[
                        0, y : y + self.rf_size, x : x + self.rf_size
                    ]

            # Reshape patches and STAs for batch dot product computation
            # patches: (batch_size, n_neurons, 1, rf_size, rf_size)
            # -> (batch_size, n_neurons, rf_size^2)
            patches_flat = patches.view(batch_size_actual, self.n_neurons, -1)
            # stas: (n_neurons, rf_size, rf_size) -> (n_neurons, rf_size^2)
            stas_flat = self.stas_tensor.view(self.n_neurons, -1)

            print(
                f"  Computing neural responses for batch {batch_idx + 1}/"
                f"{n_batches}..."
            )
            for i in range(batch_size_actual):
                # Compute all dot products for this image at once
                # patches_flat[i]: (n_neurons, rf_size^2)
                # stas_flat: (n_neurons, rf_size^2)
                # dot: (n_neurons,)
                dot = torch.mean(patches_flat[i] * stas_flat, dim=1)
                dot_products[start_idx + i] = dot.cpu().numpy()

                # Use a steeper non-linearity for more sparsity
                response = F.elu(dot - thresholds) + 1
                response = max_rates * response

                # Add noise if specified
                if noise_level > 0:
                    noise_factor = 1.0 + 0.2 * noise_level * torch.rand(
                        self.n_neurons, device=self.device
                    )
                    response = response * noise_factor

                # Add baseline firing rate
                response = response + baselines

                # Final firing rate with physiological limits
                firing_rate = torch.clamp(
                    response, min=0.0
                )  # First clamp to 0
                firing_rate = torch.minimum(
                    firing_rate, max_rates
                )  # Then clamp to max_rates

                firing_rates[start_idx + i] = firing_rate.cpu().numpy()

            # Clear GPU memory after each batch
            del patches, patches_flat
            torch.cuda.empty_cache()

        return firing_rates, dot_products, np.zeros_like(firing_rates)
