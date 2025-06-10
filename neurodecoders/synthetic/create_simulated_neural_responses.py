import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
import os
from image_datasets import ImageDataset
from sta import STA
from simulate_response import SimulateResponse
import datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_output(images, responses, stas, coords, filename):
    np.savez(filename,
             images=images.cpu().numpy(),
             responses=responses,
             stas=stas,
             rf_coords=coords)

def plot_sta_and_spikes(images, responses, dot_products, stas, coords, n_plot_images=5, n_top_neurons=5):
    n_images, n_neurons = responses.shape
    
    # First sort images by their maximum firing rate
    image_max_responses = np.max(responses, axis=1)
    image_sort_idx = np.argsort(image_max_responses)[::-1]  # Descending order
    # Convert tensor to numpy for sorting
    images_np = images.cpu().numpy()
    sorted_images = torch.from_numpy(images_np[image_sort_idx]).to(images.device)
    sorted_responses = responses[image_sort_idx]
    sorted_dot_products = dot_products[image_sort_idx]
    
    # Then sort neurons by their response to the highest responding image
    neuron_sort_idx = np.argsort(sorted_responses[0])[::-1]  # Descending order
    # Take only top n_top_neurons
    neuron_sort_idx = neuron_sort_idx[:n_top_neurons]
    sorted_responses = sorted_responses[:, neuron_sort_idx]
    sorted_dot_products = sorted_dot_products[:, neuron_sort_idx]
    sorted_stas = stas[neuron_sort_idx]
    sorted_coords = coords[neuron_sort_idx]
    
    # Find global max for y-axis scaling
    y_max = max(np.max(sorted_responses), np.max(sorted_dot_products))
    
    # Create first figure for images and responses
    fig1 = plt.figure(figsize=(20, 10))  # Wider figure for side-by-side layout
    gs = fig1.add_gridspec(n_plot_images, 4)  # 4 columns: 2 for first set, 2 for second set
    
    # Colors for different neurons
    colors = plt.cm.tab10(np.linspace(0, 1, n_top_neurons))
    
    # Plot responses for each image in first panel (left side)
    for i in range(n_plot_images):
        # Plot image with receptive fields
        ax_img = fig1.add_subplot(gs[i, 0])
        img_raw = sorted_images[i][0].cpu().numpy()
        # Convert from [-1, 1] to [0, 1] for display
        img_raw = (img_raw + 1) / 2
        img_raw = np.clip(img_raw, 0, 1)
        ax_img.imshow(img_raw, cmap='gray')
        
        # Add receptive field rectangles for all neurons
        for n in range(n_top_neurons):
            x, y = sorted_coords[n]
            rf_size = 63
            rect = Rectangle((x, y), rf_size, rf_size, 
                           linewidth=3.0,
                           edgecolor=colors[n], 
                           facecolor='none', 
                           alpha=0.8)
            ax_img.add_patch(rect)
        ax_img.axis('off')
        ax_img.set_title(f"Image {image_sort_idx[i]} (Max Response: {image_max_responses[image_sort_idx[i]]:.2f})")
        
        # Plot firing rates for all neurons
        ax_resp = fig1.add_subplot(gs[i, 1])
        # Plot dots for firing rates
        ax_resp.scatter(range(n_top_neurons), sorted_responses[i], 
                       c=colors[:n_top_neurons], s=100, 
                       label='Firing Rate')
        # Plot dots for dot products
        ax_resp.scatter(range(n_top_neurons), sorted_dot_products[i], 
                       c=colors[:n_top_neurons], s=100, marker='x',
                       label='Dot Product')
        
        ax_resp.set_title(f"Neural Responses to Image {image_sort_idx[i]}")
        ax_resp.set_xlabel("Neuron (sorted by response to highest image)")
        ax_resp.set_ylabel("Response")
        ax_resp.set_xticks(range(n_top_neurons))
        ax_resp.set_ylim(0, y_max)  # Set consistent y-axis limit
        # Only show legend for the first plot to avoid overcrowding
        if i == 0:
            ax_resp.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add a title for the second panel
    fig1.text(0.75, 0.95, "Additional Images", ha='center', fontsize=12)
    
    # Plot responses for additional images in second panel (right side)
    for i in range(n_plot_images):
        # Plot image with receptive fields
        ax_img = fig1.add_subplot(gs[i, 2])
        img_raw = sorted_images[i + n_plot_images][0].cpu().numpy()
        # Convert from [-1, 1] to [0, 1] for display
        img_raw = (img_raw + 1) / 2
        img_raw = np.clip(img_raw, 0, 1)
        ax_img.imshow(img_raw, cmap='gray')
        
        # Add receptive field rectangles for all neurons
        for n in range(n_top_neurons):
            x, y = sorted_coords[n]
            rf_size = 63
            rect = Rectangle((x, y), rf_size, rf_size, 
                           linewidth=3.0,
                           edgecolor=colors[n], 
                           facecolor='none', 
                           alpha=0.8)
            ax_img.add_patch(rect)
        ax_img.axis('off')
        ax_img.set_title(f"Image {image_sort_idx[i + n_plot_images]} (Max Response: {image_max_responses[image_sort_idx[i + n_plot_images]]:.2f})")
        
        # Plot firing rates for all neurons
        ax_resp = fig1.add_subplot(gs[i, 3])
        # Plot dots for firing rates
        ax_resp.scatter(range(n_top_neurons), sorted_responses[i + n_plot_images], 
                       c=colors[:n_top_neurons], s=100, 
                       label='Firing Rate')
        # Plot dots for dot products
        ax_resp.scatter(range(n_top_neurons), sorted_dot_products[i + n_plot_images], 
                       c=colors[:n_top_neurons], s=100, marker='x',
                       label='Dot Product')
        
        ax_resp.set_title(f"Neural Responses to Image {image_sort_idx[i + n_plot_images]}")
        ax_resp.set_xlabel("Neuron (sorted by response to highest image)")
        ax_resp.set_ylabel("Response")
        ax_resp.set_xticks(range(n_top_neurons))
        ax_resp.set_ylim(0, y_max)  # Set consistent y-axis limit
    
    plt.tight_layout()
    
    # Create second figure for STAs
    fig2 = plt.figure(figsize=(15, 3))
    gs_sta = fig2.add_gridspec(1, n_top_neurons)
    
    # Plot all STAs in a grid
    for i in range(n_top_neurons):
        ax_sta = fig2.add_subplot(gs_sta[0, i])
        sta = sorted_stas[i]
        # Ensure STA is 2D for plotting
        if len(sta.shape) == 3:
            img_sta = sta[0]  # Take first channel if 3D
        else:
            img_sta = sta
        # Convert from [-1, 1] to [0, 1] for display
        img_sta = (img_sta + 1) / 2
        ax_sta.imshow(img_sta, cmap='gray')
        for spine in ax_sta.spines.values():
            spine.set_edgecolor(colors[i])
            spine.set_linewidth(3)
        ax_sta.set_title(f"N{neuron_sort_idx[i]}", color=colors[i], fontsize=8)
        ax_sta.axis('off')
    
    plt.tight_layout()
    
    return fig1, fig2


def main():
    n_images = 1000
    n_neurons = 1000

    print("Loading data and model...")
    images = ImageDataset().get_data("mnist", n_images)
    stas = STA().get_simulated_sta("perlin_noise_patterns,11,11")

    print("Generating responses...")
    simulator = SimulateResponse(device, images, stas, n_neurons)
    firing_rates, dot_products = simulator.simulate_neural_responses()

    print("Plotting example results...")
    fig1, fig2 = plot_sta_and_spikes(images, firing_rates, dot_products, simulator.selected_stas, simulator.rf_coords)

    print("Saving dataset...")
    os.makedirs("output", exist_ok=True)
    save_output(images, firing_rates, simulator.selected_stas, simulator.rf_coords, 
                f"output/simulated_neural_data_{n_neurons}neurons_{n_images}images_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npz")
    print("Done.")

if __name__ == "__main__":
    main()
