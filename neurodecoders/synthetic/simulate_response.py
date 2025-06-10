import torch
import numpy as np
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
        self.stas_tensor = torch.tensor(self.selected_stas, dtype=torch.float32).to(self.device)

        self.rf_coords = np.random.randint(0, 224 - self.rf_size, size=(n_neurons, 2))

    def get_receptive_field(self, image, x, y, size):
        return image[:, y:y+size, x:x+size]

    def simulate_neural_responses(self, noise_level=0.1):
        firing_rates = np.zeros((self.n_images, self.n_neurons))
        dot_products = np.zeros((self.n_images, self.n_neurons))
        adaptation_states = np.zeros((self.n_images, self.n_neurons))  # Track adaptation state
        
        # Generate neuron-specific parameters with physiological constraints
        # Baseline rates: mostly very low (0.1-2 Hz), some higher
        baselines = torch.exp(torch.randn(self.n_neurons, device=self.device) * 0.3) * 0.2
        
        # Thresholds: higher thresholds to create more sparsity
        thresholds = 0.2 + 0.3 * torch.rand(self.n_neurons, device=self.device)
        
        # Maximum firing rates: respecting physiological limits
        # Most neurons max out at 100-200 Hz, with some exceptions
        max_rates = torch.exp(torch.randn(self.n_neurons, device=self.device) * 0.3) * 100
        
        # Initialize adaptation state (start at 1.0, will decrease with adaptation)
        adaptation_state = torch.ones(self.n_neurons, device=self.device)
        
        for i in tqdm(range(self.n_images), desc="Simulating neural responses"):
            image = self.images[i].to(self.device)
            
            # Slow recovery of adaptation (increase back towards 1.0)
            adaptation_state = 1.0 - (1.0 - adaptation_state) * torch.exp(torch.tensor(-0.1, device=self.device))
            
            for n in range(self.n_neurons):
                x, y = self.rf_coords[n]
                patch = self.get_receptive_field(image, x, y, self.rf_size).unsqueeze(0)
                sta = self.stas_tensor[n].unsqueeze(0)
                sta_flat = sta.reshape(-1)
                patch_flat = patch.reshape(-1)
                
                # Compute dot product
                dot = torch.sum(torch.abs(patch_flat * sta_flat)) / len(patch_flat)
                dot_products[i, n] = dot.item()
                
                # Apply threshold and non-linearity
                # Use a steeper non-linearity for more sparsity
                response = F.relu(dot - thresholds[n])
                response = max_rates[n] * response
                
                # Apply adaptation from previous response (decrease response)
                response = response * adaptation_state[n]
                
                # Add baseline 
                response = response + baselines[n]
                
                # Add noise
                poisson_noise = torch.sqrt(response) * torch.randn(1, device=self.device) * noise_level
                
                
                # Final firing rate
                firing_rate = torch.clamp(response + poisson_noise, min=0, max=max_rates[n]).item()
                
                # Update adaptation based on current response for next image
                # Decrease adaptation state (stronger adaptation for higher responses)
                adaptation_factor = 0.5 * (firing_rate / max_rates[n])  # How much to decrease by
                adaptation_state[n] = adaptation_state[n] * (1.0 - adaptation_factor)  # Decrease adaptation state
                
                firing_rates[i, n] = firing_rate
                adaptation_states[i, n] = adaptation_state[n].item()  # Save adaptation state
                    
        return firing_rates, dot_products, adaptation_states
    
    def spike_train_from_firing_rate(self, firing_rate, sampling_rate, timepoints):
        prob = firing_rate / sampling_rate
        spikes = np.zeros(timepoints)
        t = 0
        refractory_bins = int(0.002 * sampling_rate)  # 2 ms
        while t < timepoints:
            if np.random.rand() < prob:
                spikes[t] = 1.0
                t += refractory_bins
            else:
                t += 1
        return spikes
