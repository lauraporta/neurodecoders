import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class SimulateResponse:
    def __init__(self, device, images_or_loader_fn, stas, n_neurons, n_images=None, image_height=32, image_width=32):
        self.device = device
        # Support both pre-loaded images (tensor) or a function that creates a DataLoader
        if callable(images_or_loader_fn):
            self.loader_fn = images_or_loader_fn
            self.images = None
            if n_images is None:
                raise ValueError("n_images must be provided when using a DataLoader function")
            self.n_images = n_images
        else:
            self.images = images_or_loader_fn
            self.loader_fn = None
            self.n_images = len(images_or_loader_fn)
        
        self.stas = stas
        self.n_neurons = n_neurons
        self.rf_size = stas.shape[1]
        self.image_height = image_height
        self.image_width = image_width

        self.selected_indices = np.random.choice(stas.shape[0], n_neurons)
        self.selected_stas = stas[self.selected_indices]
        self.stas_tensor = torch.tensor(
            self.selected_stas, dtype=torch.float32
        ).to(self.device)

        # Place RFs randomly within image bounds
        self.rf_coords = np.column_stack([
            np.random.randint(0, max(1, self.image_width - self.rf_size), size=n_neurons),
            np.random.randint(0, max(1, self.image_height - self.rf_size), size=n_neurons)
        ])

    def estimate_memory_usage(self, batch_size=100, neuron_batch_size=1000):
        """
        Estimate GPU memory usage for the simulation.

        Args:
            batch_size: Number of images to process in each batch
            neuron_batch_size: Number of neurons to process in each batch

        Returns:
            dict: Memory usage estimates in GB
        """
        # Memory per patch: batch_size * neuron_batch_size * 1 * rf_size * rf_size * 4 bytes
        patch_memory_gb = (
            batch_size * neuron_batch_size * 1 * self.rf_size * self.rf_size * 4
        ) / (1024**3)

        # Memory for other tensors (approximate)
        other_tensors_gb = (
            batch_size * neuron_batch_size * 4 * 4  # Various intermediate tensors
        ) / (1024**3)

        total_memory_gb = patch_memory_gb + other_tensors_gb

        return {
            "patch_memory_gb": patch_memory_gb,
            "other_tensors_gb": other_tensors_gb,
            "total_memory_gb": total_memory_gb,
        }

    def simulate_neural_responses_vectorized(
        self, noise_level=1, batch_size=100, neuron_batch_size=1000
    ):
        """
        Memory-efficient version of neural response simulation using batch
        processing for both images and neurons.
        
        Args:
            noise_level: Level of Gaussian noise to add
            batch_size: Number of images to process in each batch (only used with DataLoader)
            neuron_batch_size: Number of neurons to process in each batch
        """
        print("Preparing data for batch computation...")

        # max firing rate is 100Hz with no variability
        max_firing_rate = 100

        # Pre-allocate output tensors on CPU (will be moved to GPU in batches)
        firing_rates = np.zeros((self.n_images, self.n_neurons))
        dot_products = np.zeros((self.n_images, self.n_neurons))
        noises = np.zeros((self.n_images, self.n_neurons))

        # Calculate number of batches for neurons
        n_neuron_batches = (self.n_neurons + neuron_batch_size - 1) // neuron_batch_size

        print(
            f"Processing {self.n_neurons} neurons in {n_neuron_batches} batches of "
            f"up to {neuron_batch_size} neurons"
        )

        if self.loader_fn is not None:
            # CRITICAL FIX: Process neuron batches in outer loop to avoid DataLoader exhaustion
            # Each neuron batch gets a fresh DataLoader with all images
            print(f"Processing images from DataLoader in batches")
            
            for neuron_batch_idx in tqdm(range(n_neuron_batches), desc="Processing neuron batches"):
                neuron_start_idx = neuron_batch_idx * neuron_batch_size
                neuron_end_idx = min((neuron_batch_idx + 1) * neuron_batch_size, self.n_neurons)
                
                # Create a fresh DataLoader for this neuron batch
                data_loader = self.loader_fn()
                
                img_start_idx = 0
                
                # Process all images for this neuron batch
                for batch_images, batch_labels in tqdm(
                    data_loader, 
                    desc=f"Images (neurons {neuron_start_idx}-{neuron_end_idx})",
                    leave=False
                ):
                    batch_size_actual = len(batch_images)
                    img_end_idx = img_start_idx + batch_size_actual
                    
                    if img_end_idx > self.n_images:
                        # Trim the batch if we have more images than needed
                        batch_size_actual = self.n_images - img_start_idx
                        batch_images = batch_images[:batch_size_actual]
                        img_end_idx = self.n_images
                    
                    # Process this image batch for current neuron batch only
                    self._process_single_neuron_batch(
                        batch_images, img_start_idx, img_end_idx,
                        neuron_start_idx, neuron_end_idx,
                        max_firing_rate, noise_level,
                        firing_rates, dot_products, noises
                    )
                    
                    # Clean up after each image batch
                    del batch_images, batch_labels
                    torch.cuda.empty_cache()
                    
                    img_start_idx = img_end_idx
                    
                    if img_start_idx >= self.n_images:
                        break
                
                # Clean up the DataLoader after finishing this neuron batch
                del data_loader
                torch.cuda.empty_cache()
        else:
            # Use pre-loaded images - process in batches
            n_image_batches = (self.n_images + batch_size - 1) // batch_size
            print(
                f"Processing {self.n_images} images in {n_image_batches} batches of "
                f"up to {batch_size} images"
            )
            
            for img_batch_idx in tqdm(range(n_image_batches), desc="Processing image batches"):
                img_start_idx = img_batch_idx * batch_size
                img_end_idx = min((img_batch_idx + 1) * batch_size, self.n_images)
                batch_images = self.images[img_start_idx:img_end_idx]
                
                # Process neurons in batches for each image batch
                self._process_image_batch(
                    batch_images, img_start_idx, img_end_idx,
                    neuron_batch_size, n_neuron_batches,
                    max_firing_rate, noise_level,
                    firing_rates, dot_products, noises
                )

        return firing_rates, dot_products, noises

    def _process_neuron_batch_for_images(
        self, batch_images, img_start_idx, img_end_idx,
        neuron_start_idx, neuron_end_idx,
        max_firing_rate, noise_level,
        firing_rates, dot_products, noises
    ):
        """
        Core processing function: process a batch of images for a specific neuron batch.
        This is the single source of truth for the computation logic.
        """
        batch_size_actual = img_end_idx - img_start_idx
        neuron_batch_size_actual = neuron_end_idx - neuron_start_idx
        
        # Use torch.no_grad() to prevent gradient accumulation
        with torch.no_grad():
            # Pre-extract receptive field patches for this image-neuron batch
            patches = torch.zeros(
                (
                    batch_size_actual,
                    neuron_batch_size_actual,
                    1,
                    self.rf_size,
                    self.rf_size,
                ),
                device=self.device,
            )

            for i in range(batch_size_actual):
                image = batch_images[i].to(self.device)
                for n in range(neuron_batch_size_actual):
                    neuron_idx = neuron_start_idx + n
                    x, y = self.rf_coords[neuron_idx]
                    patches[i, n, 0] = image[
                        0, y : y + self.rf_size, x : x + self.rf_size
                    ]

            # Flatten patches and STAs for this neuron batch
            patches_flat = patches.view(batch_size_actual, neuron_batch_size_actual, -1)
            stas_flat = self.stas_tensor[neuron_start_idx:neuron_end_idx].view(
                neuron_batch_size_actual, -1
            )

            # Compute neural responses for this batch
            for i in range(batch_size_actual):
                dot = torch.sum(patches_flat[i] * stas_flat, dim=1)
                dot_products[img_start_idx + i, neuron_start_idx:neuron_end_idx] = dot.cpu().numpy()
                response = F.elu(dot)

                #  normalise to max firing rate
                response = response / (response.max() + 1e-6) * max_firing_rate

                gaussian_noise = torch.randn(
                    neuron_batch_size_actual, device=self.device
                )
                noise = gaussian_noise * noise_level
                noises[img_start_idx + i, neuron_start_idx:neuron_end_idx] = noise.cpu().numpy()
                response += noise

                firing_rate = torch.clamp(response, min=0.0)

                firing_rates[img_start_idx + i, neuron_start_idx:neuron_end_idx] = firing_rate.cpu().numpy()

            # Clear GPU memory
            del patches, patches_flat, stas_flat, image
            torch.cuda.empty_cache()

    def _process_single_neuron_batch(
        self, batch_images, img_start_idx, img_end_idx,
        neuron_start_idx, neuron_end_idx,
        max_firing_rate, noise_level,
        firing_rates, dot_products, noises
    ):
        """
        Process a batch of images for a SINGLE neuron batch.
        Delegates to the core processing function.
        """
        self._process_neuron_batch_for_images(
            batch_images, img_start_idx, img_end_idx,
            neuron_start_idx, neuron_end_idx,
            max_firing_rate, noise_level,
            firing_rates, dot_products, noises
        )

    def _process_image_batch(
        self, batch_images, img_start_idx, img_end_idx,
        neuron_batch_size, n_neuron_batches,
        max_firing_rate, noise_level,
        firing_rates, dot_products, noises
    ):
        """
        Process a batch of images across ALL neuron batches.
        Loops through neuron batches and delegates to the core processing function.
        """
        for neuron_batch_idx in range(n_neuron_batches):
            neuron_start_idx = neuron_batch_idx * neuron_batch_size
            neuron_end_idx = min((neuron_batch_idx + 1) * neuron_batch_size, self.n_neurons)
            
            self._process_neuron_batch_for_images(
                batch_images, img_start_idx, img_end_idx,
                neuron_start_idx, neuron_end_idx,
                max_firing_rate, noise_level,
                firing_rates, dot_products, noises
            )
