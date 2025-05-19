# ## Simulated Neural Response Generator

# ### Goal
# Create a Python script that generates simulated neural responses to natural images. These synthetic neural responses will be used later to train separate encoder and decoder models.

# ### Overview
# This script will:
# 1. Load a dataset of natural images (CIFAR-10).
# 2. Extract Gabor-like filters from the first convolutional layer of a pretrained CNN (Alexnet) to use as synthetic "spike-triggered averages" (STAs).
# 3. Generate simulated neural responses using a biologically inspired encoding model:
#    - Each neuron's response is modeled as the dot product between the image and its STA,
#    - Passed through a ReLU nonlinearity,
#    - With added Gaussian noise.
# 4. Randomly assign receptive field preferences:
#    - Each neuron only responds to a random subset of images (simulating sparsity).
#    - Some filters may not have any responsive neurons.
# 5. Save the resulting dataset: For each image → a vector of synthetic neural responses.

# ### Components
# **Input:**
# - Dataset of images (CIFAR-10, resized to match the CNN input size if needed)
# - Pretrained CNN (`torchvision.models.alexnet(pretrained=True)`)

# **Output:**
# - A matrix of shape `[n_images, n_neurons]` containing simulated neural responses
# - Optionally: save the STAs used and neuron metadata (e.g., index of active filters)

# ### Encoding Model
# For each neuron:
# ```python
# L_i = ReLU(⟨image, STA_i⟩ + noise)
# ```
# Where:
# - `STA_i` is a 2D or 3D filter (from the CNN's first conv layer)
# - `⟨.,.⟩` is the dot product (can be implemented via convolution)
# - `ReLU(x) = max(0, x)`
# - `noise ~ Normal(0, σ)` (e.g., σ = 0.1 × response magnitude)

# Optional enhancements:
# - Simulate spatial localization by applying each filter at fixed receptive field positions (e.g., sample conv activations at specific locations).

# ### Parameters
# Can be passed via argparse or hardcoded initially:
# - `n_neurons`: Number of synthetic neurons to simulate
# - `filter_selection`: Method to sample STAs (e.g., randomly select filters from `conv1`)
# - `noise_level`: Standard deviation of the Gaussian noise
# - `activation_threshold`: (optional) Minimum value for a neuron to be considered active for an image
# - `image_dataset`: Name or path of the dataset to use
# - `output_path`: Where to save the `.npz` or `.h5` output

# ### Output Format
# Save:
# ```python
# {
#   'images': np.ndarray,            # shape: [n_images, C, H, W]
#   'responses': np.ndarray,         # shape: [n_images, n_neurons]
#   'stas': np.ndarray,              # shape: [n_neurons, C, kH, kW]
#   'neuron_metadata': dict          # (optional): includes things like which conv filter and spatial position each neuron uses
# }
# ```

# ### Implementation Hints
# - Use `torchvision.models` to load a CNN and extract first conv layer filters.
# - Normalize STAs (e.g. zero mean, unit norm) to control the dynamic range of responses.
# - Use `torch.utils.data.DataLoader` to batch-load images.
# - Convert everything to NumPy arrays at the end for easy saving and compatibility.

# ### Deliverables
# - A single script (e.g., `generate_simulated_responses.py`) that:
#   - Loads dataset and CNN
#   - Extracts and prepares STAs
#   - Generates noisy ReLU-based responses
#   - Saves the final dataset for downstream encoder/decoder training



import torch
import torchvision
import torchvision.transforms as transforms
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_cifar10():
    """Load and preprocess CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to AlexNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                            shuffle=True, num_workers=2)
    return trainloader

def extract_stas(model):
    """Extract first convolutional layer filters as STAs."""
    # Get the first conv layer weights
    conv1_weights = model.features[0].weight.data.cpu().numpy()
    return conv1_weights

def generate_spike_trains(firing_rate, duration=1.0, sampling_rate=10, n_trials=10):
    """
    Generate spike trains based on a firing rate.
    
    Args:
        firing_rate: Firing rate in Hz
        duration: Duration of spike train in seconds
        sampling_rate: Sampling rate in Hz
        n_trials: Number of spike trains to generate
    
    Returns:
        spike_trains: Binary array of shape (n_trials, n_timepoints)
    """
    n_timepoints = int(duration * sampling_rate)
    spike_trains = []
    
    for _ in range(n_trials):
        # Convert firing rate to probability per time bin
        prob_per_bin = firing_rate / sampling_rate
        
        # Generate binary spike train using numpy instead of torch
        spike_train = (np.random.rand(n_timepoints) < prob_per_bin).astype(float)
        spike_trains.append(spike_train)
    
    return np.stack(spike_trains)

def generate_spike_trains_for_image(firing_rate, duration=1.0, sampling_rate=10, n_trials=10):
    """
    Generate spike trains for a single image's firing rate.
    This function is used for parallel processing.
    """
    return generate_spike_trains(firing_rate, duration, sampling_rate, n_trials)

def generate_poisson_spikes(firing_rate, duration=1.0, sampling_rate=1000, n_trials=1):
    """
    Generate Poisson spike trains based on a firing rate.
    
    Args:
        firing_rate: Firing rate in Hz
        duration: Duration of spike train in seconds
        sampling_rate: Sampling rate in Hz
        n_trials: Number of spike trains to generate
    
    Returns:
        spike_trains: Binary array of shape (n_trials, n_timepoints)
    """
    n_timepoints = int(duration * sampling_rate)
    spike_trains = []
    
    for _ in range(n_trials):
        # Convert firing rate to probability per time bin
        prob_per_bin = firing_rate / sampling_rate
        
        # Generate binary spike train using numpy
        spike_train = (np.random.rand(n_timepoints) < prob_per_bin).astype(float)
        spike_trains.append(spike_train)
    
    return np.stack(spike_trains)

def generate_neural_responses(images, stas, n_neurons, noise_level=0.1, image_duration=0.2, sampling_rate=1000, n_trials=10):
    """
    Generate simulated neural responses to a sequence of images.
    
    Args:
        images: Input images tensor [batch_size, channels, height, width]
        stas: Spike-triggered averages [n_filters, channels, height, width]
        n_neurons: Number of neurons to simulate
        noise_level: Standard deviation of Gaussian noise
        image_duration: Duration of each image presentation in seconds
        sampling_rate: Sampling rate in Hz
        n_trials: Number of trials to generate
    
    Returns:
        responses: Neural responses [n_trials, n_neurons, total_timepoints]
        selected_stas: STAs used for each neuron [n_neurons, channels, height, width]
        image_sequence: Array indicating which image is shown at each timepoint
    """
    batch_size = images.shape[0]
    n_filters = stas.shape[0]
    
    # Calculate total duration and number of timepoints
    total_duration = batch_size * image_duration
    n_timepoints = int(total_duration * sampling_rate)
    points_per_image = int(image_duration * sampling_rate)
    
    # Create image sequence array
    image_sequence = np.repeat(np.arange(batch_size), points_per_image)
    
    # Randomly select STAs for each neuron
    selected_filter_indices = np.random.randint(0, n_filters, size=n_neurons)
    selected_stas = stas[selected_filter_indices]
    
    # Convert to torch tensors
    selected_stas_tensor = torch.from_numpy(selected_stas).float().to(device)
    images_tensor = images.to(device)
    
    # Initialize response array
    responses = np.zeros((n_trials, n_neurons, n_timepoints))
    
    # # Create a pool of workers for parallel processing
    # n_cores = mp.cpu_count()
    # pool = mp.Pool(processes=n_cores)
    
    for trial in range(n_trials):
        print(f"\nGenerating trial {trial + 1}/{n_trials}")
        for i in tqdm(range(n_neurons), desc=f"Processing neurons for trial {trial + 1}"):
            # Get neuron's STA
            sta = selected_stas_tensor[i:i+1]
            
            # Compute response for each image
            image_responses = []
            for img_idx in range(batch_size):
                # Convolve image with STA
                conv_output = F.conv2d(images_tensor[img_idx:img_idx+1], 
                                     sta,
                                     padding='same')
                
                # Apply ReLU and add noise
                response = F.relu(conv_output)
                noise = torch.randn_like(response) * noise_level * torch.mean(response)
                response = response + noise
                
                # Take spatial average
                response = torch.mean(response, dim=(2, 3))
                
                # Convert to firing rate (1-200 Hz)
                firing_rate = 1 + 199 * (response - response.min()) / (response.max() - response.min())
                
                # Generate spike train for this image
                spike_train = generate_poisson_spikes(
                    firing_rate.item(),
                    duration=image_duration,
                    sampling_rate=sampling_rate,
                    n_trials=1
                )[0]  # Take first (and only) trial
                
                image_responses.append(spike_train)
            
            # Concatenate responses for all images
            responses[trial, i] = np.concatenate(image_responses)
    
    # # Close the pool
    # pool.close()
    # pool.join()
    
    return responses, selected_stas, image_sequence

def plot_neurons_responses(stas, spike_trains, image_sequence, n_neurons=4, n_trials=5, image_duration=0.2, sampling_rate=1000):
    """
    Plot multiple neurons' STAs and their spike trains in response to the image sequence.
    Selects neurons that respond most strongly.
    
    Args:
        stas: Array of STAs [n_neurons, channels, height, width]
        spike_trains: Array of spike trains [n_trials, n_neurons, n_timepoints]
        image_sequence: Array indicating which image is shown at each timepoint
        n_neurons: Number of neurons to plot
        n_trials: Number of trials to plot per neuron
        image_duration: Duration of each image presentation in seconds
        sampling_rate: Sampling rate in Hz
    """
    # Calculate mean firing rate for each neuron
    mean_firing_rates = np.mean(spike_trains, axis=(0, 2))  # Average across trials and time
    # Select top n_neurons with highest firing rates
    top_neuron_indices = np.argsort(mean_firing_rates)[-n_neurons:][::-1]
    
    # Create figure with n_neurons rows and 2 columns
    fig, axes = plt.subplots(n_neurons, 2, figsize=(15, 3*n_neurons))
    
    # Calculate time points for x-axis
    time_points = np.arange(spike_trains.shape[2]) / sampling_rate
    
    for i, neuron_idx in enumerate(top_neuron_indices):
        # Plot STA
        sta = stas[neuron_idx]
        if sta.shape[0] == 3:
            sta = (sta - sta.min()) / (sta.max() - sta.min())
            sta = np.transpose(sta, (1, 2, 0))
        else:
            sta = sta[0]
            sta = (sta - sta.min()) / (sta.max() - sta.min())
        
        axes[i, 0].imshow(sta, cmap='gray' if sta.ndim == 2 else None)
        axes[i, 0].set_title(f'Neuron {neuron_idx}\nMean rate: {mean_firing_rates[neuron_idx]:.1f} Hz')
        axes[i, 0].axis('off')
        
        # Plot spike trains
        for trial in range(n_trials):
            spikes = spike_trains[trial, neuron_idx]
            spike_times = np.where(spikes == 1)[0] / sampling_rate
            axes[i, 1].vlines(spike_times, trial, trial + 0.8, colors='black')
        
        # Add vertical lines to indicate image boundaries
        for img_idx in range(len(np.unique(image_sequence))):
            axes[i, 1].axvline(x=img_idx * image_duration, color='r', linestyle='--', alpha=0.3)
        
        axes[i, 1].set_yticks(range(n_trials))
        if i == n_neurons - 1:  # Only show xlabel for bottom plot
            axes[i, 1].set_xlabel('Time (seconds)')
        axes[i, 1].set_ylabel('Trial')
        if i == 0:  # Only show title for top plot
            axes[i, 1].set_title('Spike Trains')
    
    plt.tight_layout()
    plt.show()

def main():
    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    trainloader = load_cifar10()
    
    # Load pretrained AlexNet
    print("Loading pretrained AlexNet...")
    model = torchvision.models.alexnet(pretrained=True)
    model = model.to(device)
    model.eval()
    
    # Extract STAs
    print("Extracting STAs...")
    stas = extract_stas(model)
    
    # Parameters
    n_neurons = 5
    noise_level = 0.1
    image_duration = 0.2  # 200ms per image
    sampling_rate = 1000  # 1kHz sampling rate
    n_trials = 2  # 10 trials
    
    # Generate responses for a batch of images
    print("Generating neural responses...")
    images, _ = next(iter(trainloader))
    images = images.to(device)
    
    responses, selected_stas, image_sequence = generate_neural_responses(
        images, stas, n_neurons, noise_level,
        image_duration=image_duration,
        sampling_rate=sampling_rate,
        n_trials=n_trials
    )
    
    # Plot multiple neurons' responses
    print("Plotting neurons' responses...")
    plot_neurons_responses(selected_stas, responses, image_sequence, 
                         n_neurons=4, n_trials=n_trials,
                         image_duration=image_duration,
                         sampling_rate=sampling_rate)
    
    # Save results
    print("Saving results...")
    np.savez('simulated_neural_responses.npz',
             images=images.cpu().numpy(),
             responses=responses,
             stas=selected_stas,
             image_sequence=image_sequence)
    
    print("Done!")

if __name__ == "__main__":
    main()

