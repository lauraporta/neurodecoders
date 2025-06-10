import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_cifar10(n_images):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=n_images, shuffle=True)
    images, _ = next(iter(loader))
    return images[:n_images]

def extract_stas(model):
    with torch.no_grad():
        weights = model.features[0].weight.data.cpu().numpy()
        weights = weights[:, :1, :, :]
        import scipy.ndimage
        resized_filters = np.array([
            scipy.ndimage.zoom(w, (1, 63 / w.shape[1], 63 / w.shape[2]), order=1) / np.linalg.norm(w)
            for w in weights
        ])
        return resized_filters

def get_receptive_field(image, x, y, size):
    return image[:, y:y+size, x:x+size]

def simulate_neural_responses(images, stas, n_neurons=10, rf_size=63, image_duration=0.2, 
                               sampling_rate=1000, n_trials=5, noise_level=0.1):
    n_images = images.shape[0]
    timepoints = int(image_duration * sampling_rate)
    responses = np.zeros((n_trials, n_images, n_neurons, timepoints))

    selected_indices = np.random.choice(stas.shape[0], n_neurons)
    selected_stas = stas[selected_indices]
    stas_tensor = torch.tensor(selected_stas, dtype=torch.float32).to(device)

    rf_coords = np.random.randint(0, 224 - rf_size, size=(n_neurons, 2))

    for trial in range(n_trials):
        for i in tqdm(range(n_images), desc=f"Trial {trial+1}"):
            image = images[i].to(device)
            for n in range(n_neurons):
                x, y = rf_coords[n]
                patch = get_receptive_field(image, x, y, rf_size).unsqueeze(0)
                sta = stas_tensor[n].unsqueeze(0)
                sta_flat = sta.reshape(-1)
                patch_flat = patch.reshape(-1)
                dot = F.relu(F.cosine_similarity(patch_flat, sta_flat, dim=0))
                noise = noise_level * dot * torch.randn(1, device=device)
                firing_rate = torch.clamp(dot + noise, min=0).item() * 200
                prob = firing_rate / sampling_rate
                spikes = np.zeros(timepoints)
                t = 0
                refractory_bins = int(0.002 * sampling_rate)  # 5 ms
                while t < timepoints:
                    if np.random.rand() < prob:
                        spikes[t] = 1.0
                        t += refractory_bins
                    else:
                        t += 1
                responses[trial, i, n] = spikes
    return responses, selected_stas, rf_coords

def save_output(images, responses, stas, coords, filename):
    np.savez(filename,
             images=images.cpu().numpy(),
             responses=responses,
             stas=stas,
             rf_coords=coords)

def plot_sta_and_spikes(images, responses, stas, coords, sampling_rate=1000, image_duration=0.2, n_plot=4):
    n_trials, n_images, n_neurons, timepoints = responses.shape
    time = np.arange(timepoints) / sampling_rate

    fig, axes = plt.subplots(n_plot + 1, n_plot + 1, figsize=(4 * (n_plot + 1), 3 * (n_plot + 1)))

    # Top row: STAs with colored borders
    colors = plt.cm.tab10(np.linspace(0, 1, n_plot))
    for j in range(n_plot):
        sta = stas[j]
        img_sta = sta[0]
        img_sta = (img_sta - img_sta.min()) / (img_sta.max() - img_sta.min())
        
        axes[0, j + 1].imshow(img_sta, cmap='gray')
        for spine in axes[0, j + 1].spines.values():
            spine.set_edgecolor(colors[j])
            spine.set_linewidth(3)
        axes[0, j + 1].set_title(f"STA {j}", color=colors[j])
        axes[0, j + 1].axis('off')

    # Determine best image per neuron
    best_img_indices = []
    for idx in range(n_plot):
        mean_firing = responses[:, :, idx, :].sum(axis=-1).mean(axis=0)
        best_img_idx = np.argmax(mean_firing)
        best_img_indices.append(best_img_idx)

    # Plot images and spike trains
    for i, img_idx in enumerate(best_img_indices):
        # Left column: image
        img_raw = images[img_idx][0].cpu().numpy()
        img_raw = (img_raw * 0.5) + 0.5
        img_raw = np.clip(img_raw, 0, 1)
        axes[i + 1, 0].imshow(img_raw, cmap='gray')
        axes[i + 1, 0].axis('off')
        axes[i + 1, 0].set_title(f"Image {img_idx}")

        # Overlay all receptive fields
        for idx in range(n_plot):
            x, y = coords[idx]
            rf_size = 63
            color = colors[idx]
            rect = Rectangle((x, y), rf_size, rf_size, linewidth=1.5, edgecolor=color, facecolor='none')
            axes[i + 1, 0].add_patch(rect)

        # Plot raster for each neuron
        for j in range(n_plot):
            ax = axes[i + 1, j + 1]
            for trial in range(n_trials):
                spikes = responses[trial, img_idx, j]
                spike_times = np.where(spikes == 1)[0] / sampling_rate
                ax.vlines(spike_times, trial, trial + 0.8, color='black')
            ax.set_title(f"Neuron {j}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Trial")

    # Remove unused top-left
    axes[0, 0].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    n_images = 1000
    n_neurons = 200
    n_trials = 1

    print("Loading data and model...")
    images = load_cifar10(n_images)
    model = torchvision.models.alexnet(pretrained=True).to(device)
    model.eval()
    stas = extract_stas(model)

    print("Generating responses...")
    responses, selected_stas, coords = simulate_neural_responses(images, stas,
                                                                 n_neurons=n_neurons,
                                                                 n_trials=n_trials,
                                                                 image_duration=1)

    print("Plotting example results...")
    plot_sta_and_spikes(images, responses, selected_stas, coords)

    print("Saving dataset...")
    os.makedirs("output", exist_ok=True)
    save_output(images, responses, selected_stas, coords, f"output/simulated_neural_data_{n_neurons}neurons_{n_trials}trials_{n_images}images.npz")
    print("Done.")

if __name__ == "__main__":
    main()
