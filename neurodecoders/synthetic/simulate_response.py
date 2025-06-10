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

        for i in tqdm(range(self.n_images), desc="Simulating neural responses"):
            image = self.images[i].to(self.device)
            for n in range(self.n_neurons):
                x, y = self.rf_coords[n]
                patch = self.get_receptive_field(image, x, y, self.rf_size).unsqueeze(0)
                sta = self.stas_tensor[n].unsqueeze(0)
                sta_flat = sta.reshape(-1)
                patch_flat = patch.reshape(-1)
                dot = F.relu(torch.sum(patch_flat * sta_flat) / len(patch_flat))
                noise = noise_level * dot * torch.randn(1, device=self.device)
                baseline = 0.01 * torch.randn(1, device=self.device)
                firing_rate = torch.clamp(dot + noise + baseline, min=0).item() * 200
                firing_rates[i, n] = firing_rate
                dot_products[i, n] = dot.item()
                    
        return firing_rates, dot_products
    
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
