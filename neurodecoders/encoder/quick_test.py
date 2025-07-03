import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from neurodecoders.encoder.encoder import SimpleEncoder


def quick_encoder_test(model_path, data_path=None):
    """Quick test to check if encoder is predicting constant values"""
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dict = torch.load(model_path, map_location=device)
    
    # Determine output neurons
    if 'model.fc.6.weight' in state_dict:
        out_neurons = state_dict['model.fc.6.weight'].shape[0]
    else:
        fc_keys = [k for k in state_dict.keys() if 'fc' in k and 'weight' in k]
        last_fc_key = sorted(fc_keys)[-1]
        out_neurons = state_dict[last_fc_key].shape[0]
    
    encoder = SimpleEncoder(out_neurons)
    
    # Handle nested model structure
    if any(k.startswith('model.') for k in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    encoder.load_state_dict(state_dict)
    encoder.to(device)
    encoder.eval()
    
    # Load data
    if data_path is None:
        model_name = os.path.basename(model_path).replace('.pth', '').strip()
        # Extract the key parts of the model name
        if 'mnist' in model_name and 'periodic_patterns' in model_name and '1000_n_neurons' in model_name:
            data_pattern = "data/synthdata_dataset-mnist_sta-periodic_patterns,11,11_n_neurons-1000_n_images-1000*.npz"
        elif 'mnist' in model_name and 'perlin_noise_patterns' in model_name and '100_n_neurons' in model_name:
            data_pattern = "data/synthdata_dataset-mnist_sta-perlin_noise_patterns,11,11_n_neurons-100_n_images-100*.npz"
        elif 'cifar10' in model_name and 'perlin_noise_patterns' in model_name and '1000_n_neurons' in model_name:
            data_pattern = "data/synthdata_dataset-cifar10_sta-perlin_noise_patterns,11,11_n_neurons-1000_n_images-1000*.npz"
        else:
            # Try a more general pattern
            data_pattern = "data/synthdata_dataset-*.npz"
        
        data_files = glob.glob(data_pattern)
        if data_files:
            # Use the most recent matching file
            data_path = max(data_files, key=os.path.getctime)
            print(f"Found data file: {data_path}")
        else:
            print("No matching data file found")
            return
    
    data = np.load(data_path)
    images = data['images']
    true_firing_rates = data['responses']
    
    print(f"Image shape: {images.shape}")
    print(f"Firing rates shape: {true_firing_rates.shape}")
    
    # Test with a few images
    test_images = images[:10]
    # Ensure correct shape: (batch, channels, height, width)
    if test_images.ndim == 3:
        test_tensor = torch.tensor(test_images[:, None, :, :], dtype=torch.float32).to(device)
    elif test_images.ndim == 4:
        test_tensor = torch.tensor(test_images, dtype=torch.float32).to(device)
    else:
        print(f"Unexpected image shape: {test_images.shape}")
        return
    
    with torch.no_grad():
        predictions = encoder(test_tensor).cpu().numpy()
    
    # Analyze predictions
    print(f"Test images: {len(test_images)}")
    print(f"Number of neurons: {predictions.shape[1]}")
    print(f"Prediction shape: {predictions.shape}")
    
    # Check for constant predictions
    neuron_stds = np.std(predictions, axis=0)
    constant_neurons = np.sum(neuron_stds < 0.1)
    
    print(f"\nNeurons with constant predictions (< 0.1 std): {constant_neurons}/{predictions.shape[1]}")
    print(f"Percentage: {constant_neurons/predictions.shape[1]*100:.1f}%")
    
    # Show statistics
    print(f"\nPrediction statistics:")
    print(f"  Mean across all neurons: {np.mean(predictions):.3f}")
    print(f"  Std across all neurons: {np.std(predictions):.3f}")
    print(f"  Min: {np.min(predictions):.3f}")
    print(f"  Max: {np.max(predictions):.3f}")
    
    # Show per-neuron statistics
    neuron_means = np.mean(predictions, axis=0)
    print(f"\nPer-neuron means:")
    print(f"  Mean: {np.mean(neuron_means):.3f}")
    print(f"  Std: {np.std(neuron_means):.3f}")
    print(f"  Min: {np.min(neuron_means):.3f}")
    print(f"  Max: {np.max(neuron_means):.3f}")
    
    # Plot histogram of neuron means
    plt.figure(figsize=(10, 6))
    plt.hist(neuron_means, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(neuron_means), color='red', linestyle='--', 
                label=f'Mean: {np.mean(neuron_means):.3f}')
    plt.xlabel('Mean Firing Rate')
    plt.ylabel('Number of Neurons')
    plt.title('Distribution of Mean Firing Rates Across Neurons')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('data/quick_test_neuron_means.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot firing rates over images for a few neurons
    plt.figure(figsize=(12, 8))
    for i in range(min(5, predictions.shape[1])):
        plt.subplot(2, 3, i+1)
        plt.plot(predictions[:, i], 'o-', alpha=0.7, label=f'Neuron {i}')
        plt.xlabel('Image Index')
        plt.ylabel('Firing Rate')
        plt.title(f'Neuron {i} - Std: {neuron_stds[i]:.3f}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/quick_test_neuron_responses.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'constant_neurons': constant_neurons,
        'total_neurons': predictions.shape[1],
        'neuron_means': neuron_means,
        'neuron_stds': neuron_stds
    }


if __name__ == "__main__":
    # Find latest encoder model
    model_files = glob.glob("data/encoder_model_*.pth")
    if model_files:
        latest_model = max(model_files, key=os.path.getctime)
        print(f"Testing model: {latest_model}")
        results = quick_encoder_test(latest_model)
    else:
        print("No encoder models found") 