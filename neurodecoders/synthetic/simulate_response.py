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

    def simulate_neural_responses_vectorized(self, noise_level=0.1):
        """
        Vectorized version of neural response simulation for GPU acceleration.
        """
        print("Preparing data for vectorized computation...")

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

        # Initialize adaptation state (start at 1.0,
        # will decrease with adaptation)
        adaptation_state = torch.ones(self.n_neurons, device=self.device)

        # Pre-allocate output tensors on GPU
        firing_rates = torch.zeros(
            (self.n_images, self.n_neurons), device=self.device
        )
        dot_products = torch.zeros(
            (self.n_images, self.n_neurons), device=self.device
        )
        adaptation_states = torch.zeros(
            (self.n_images, self.n_neurons), device=self.device
        )

        # Pre-extract all receptive field patches for all images and neurons
        print("Extracting receptive field patches...")
        patches = torch.zeros(
            (self.n_images, self.n_neurons, 1, self.rf_size, self.rf_size),
            device=self.device,
        )

        for i in tqdm(range(self.n_images), desc="Extracting patches"):
            image = self.images[i].to(self.device)
            for n in range(self.n_neurons):
                x, y = self.rf_coords[n]
                patches[i, n, 0] = image[
                    0, y : y + self.rf_size, x : x + self.rf_size
                ]

        # Reshape patches and STAs for batch dot product computation
        # patches: (n_images, n_neurons, 1, rf_size, rf_size)
        # -> (n_images, n_neurons, rf_size^2)
        patches_flat = patches.view(self.n_images, self.n_neurons, -1)
        # stas: (n_neurons, rf_size, rf_size) -> (n_neurons, rf_size^2)
        stas_flat = self.stas_tensor.view(self.n_neurons, -1)

        print("Computing neural responses...")
        for i in tqdm(range(self.n_images), desc="Simulating responses"):
            # Slow recovery of adaptation (increase back towards 1.0)
            adaptation_state = 1.0 - (1.0 - adaptation_state) * torch.exp(
                torch.tensor(-0.1, device=self.device)
            )

            # Compute all dot products for this image at once
            # patches_flat[i]: (n_neurons, rf_size^2)
            # stas_flat: (n_neurons, rf_size^2)
            # dot: (n_neurons,)
            dot = torch.mean(patches_flat[i] * stas_flat, dim=1)
            dot_products[i] = dot

            # Use a steeper non-linearity for more sparsity
            response = F.elu(dot - thresholds) + 1
            response = max_rates * response

            # Apply adaptation from previous response (decrease response)
            response = response * adaptation_state

            # Add noise before baseline
            # Subtle multiplicative noise: jitter response by up to ±10%
            if noise_level > 0:
                noise_factor = 1.0 + 0.2 * noise_level * torch.rand(
                    self.n_neurons, device=self.device
                )
                response = response * noise_factor

            # Add baseline firing rate
            response = response + baselines

            # Final firing rate
            firing_rate = torch.clamp(response, min=0.0)  # First clamp to 0
            firing_rate = torch.minimum(
                firing_rate, max_rates
            )  # Then clamp to max_rates

            # Update adaptation based on current response for next image
            # Decrease adaptation state (
            # stronger adaptation for higher responses)
            adaptation_factor = 0.1 * (firing_rate / max_rates)
            adaptation_state = adaptation_state * (1.0 - adaptation_factor)

            firing_rates[i] = firing_rate
            adaptation_states[i] = adaptation_state

        # Move results back to CPU and convert to numpy
        return (
            firing_rates.cpu().numpy(),
            dot_products.cpu().numpy(),
            adaptation_states.cpu().numpy(),
        )
