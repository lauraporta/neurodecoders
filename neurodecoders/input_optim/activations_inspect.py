"""
Activation inspection utility for a trained encoder.

This script mirrors the loading logic used by `mlflow_run.py` /
`optimizer.py` to load an encoder model (via `load_encoder_from_mlflow`) and
the original dataset images. It runs a small set of images through the
encoder, captures activations from the first few layers using forward hooks,
and saves per-layer activation visualizations and numpy dumps.

Usage (rough):
    python neurodecoders/input_optim/activations_inspect.py --model-id <run-id> --image-ids 0 1 2

The file is organized with %## section markers for quick navigation.
"""

#%%

from __future__ import annotations

import os
from typing import Dict, List, Tuple
import gc

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import mlflow

from neurodecoders.input_optim.optimizer import load_encoder_from_mlflow
from neurodecoders.input_optim.mlflow_run import _load_original_images_from_dataset
from neurodecoders.paths import get_path

#%% 
# ## Imports and helpers


def _to_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


#%% ## Hook registration and capture


class ActivationGrabber:
    """Registers forward hooks and stores activations."""

    def __init__(self, model: nn.Module, max_layers: int = 6, capture_linear: bool = True) -> None:
        self.model = model
        self.handles = []
        self.activations: Dict[str, torch.Tensor] = {}
        self.max_layers = max_layers
        self.capture_linear = capture_linear

    def _hook(self, name: str):
        def fn(module, inp, outp):
            # detach to avoid keeping computational graph
            try:
                self.activations[name] = outp.detach().cpu()
            except Exception:
                self.activations[name] = outp.cpu()

        return fn

    def register(self) -> None:
        # Register on conv/linear layers
        # If capture_linear is True, prioritize capturing all linear layers
        cnt = 0
        linear_cnt = 0
        
        for n, m in self.model.named_modules():
            # Check if we should capture this layer
            is_target_layer = isinstance(m, (nn.Conv2d, nn.Linear, nn.ReLU, nn.BatchNorm2d))
            
            if is_target_layer:
                # Always capture Linear layers if capture_linear is True
                if isinstance(m, nn.Linear) and self.capture_linear:
                    name = n if n else f"Linear_{linear_cnt}"
                    handle = m.register_forward_hook(self._hook(name))
                    self.handles.append(handle)
                    linear_cnt += 1
                elif cnt < self.max_layers:
                    name = n if n else m.__class__.__name__ + f"_{cnt}"
                    handle = m.register_forward_hook(self._hook(name))
                    self.handles.append(handle)
                    cnt += 1

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


#%% ## Visualization helpers


def plot_activation_map(act: np.ndarray, out_path: str, title: str = "") -> None:
    """Plot a single activation map (H x W) or a grid when channels > 1."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if act.ndim == 2:
        plt.figure(figsize=(4, 4))
        plt.imshow(act, cmap="viridis")
        plt.colorbar()
        plt.title(title)
        plt.axis("off")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    # act has shape (C, H, W) or (N, C, H, W)
    if act.ndim == 3:
        C, H, W = act.shape
        # limit number of channels to plot
        n_show = min(16, C)
        ncols = 4
        nrows = (n_show + ncols - 1) // ncols
        plt.figure(figsize=(ncols * 2, nrows * 2))
        for i in range(n_show):
            plt.subplot(nrows, ncols, i + 1)
            plt.imshow(act[i], cmap="viridis")
            plt.axis("off")
        plt.suptitle(title)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    # act has shape (N, C, H, W) - take first sample
    if act.ndim == 4:
        act = act[0]
        plot_activation_map(act, out_path, title)
        return

    # fallback
    np.save(out_path + ".npy", act)


#%% ## Main routine


def run_inspect(
    model_id: str,
    image_ids: List[int] = [0],
    output_dir: str | None = None,
    max_layers: int = 6,
    device: torch.device | None = None,
    save_plots: bool = True,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Notebook-friendly entrypoint.

    Args:
        model_id: MLflow model/run ID to load the encoder.
        image_ids: List of image indices to load from the original dataset.
        output_dir: Directory to save plots/numpy; defaults to workspace/plots/activations.
        max_layers: Number of layers to attach hooks to.
        device: torch device to use; if None, auto-selects.
        save_plots: If True, save activation plots to disk. If False, only return data.

    Returns:
        Tuple of (activations_dict, encoder_output, images_array)
        - activations_dict: mapping layer_name -> numpy array (activation)
        - encoder_output: numpy array of model outputs
        - images_array: numpy array of input images (N, C, H, W)
    """
    if device is None:
        device = _to_device()

    # Use environment variable for MLflow tracking URI (set by launch_apps.sh)
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        print(f"Using MLflow tracking URI from environment: {env_uri}")
        mlflow.set_tracking_uri(env_uri)
    else:
        raise ValueError("MLFLOW_TRACKING_URI not set in environment.")
       
    out_dir = output_dir or get_path("workspace/plots/activations")
    if save_plots:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Loading encoder model: {model_id}")
    encoder = load_encoder_from_mlflow(model_id)
    encoder = encoder.to(device)
    encoder.eval()

    print("Loading original images from dataset")
    original_images = _load_original_images_from_dataset(model_id, image_ids)

    # Ensure images are torch tensors with correct shape (N, C, H, W)
    imgs = np.stack(original_images, axis=0)
    if imgs.ndim == 3:
        # (N, H, W) -> (N, 1, H, W)
        imgs = imgs[:, None, ...]

    imgs_t = torch.tensor(imgs.astype(np.float32)).to(device)

    grabber = ActivationGrabber(encoder, max_layers=max_layers, capture_linear=True)
    grabber.register()

    with torch.no_grad():
        out = encoder(imgs_t)

    out_np = out.detach().cpu().numpy()

    activations_np: Dict[str, np.ndarray] = {}
    for name, act in grabber.activations.items():
        act_np = act.numpy()
        activations_np[name] = act_np
        if save_plots:
            fname = f"activations_{name.replace('.', '_')}.png"
            fpath = os.path.join(out_dir, fname)
            print(f"Saving activation plot for {name} -> {fpath}")
            try:
                plot_activation_map(act_np, fpath, title=name)
            except Exception as e:
                print(f"Could not plot {name}: {e}. Saving raw numpy instead.")
                np.save(os.path.join(out_dir, f"activations_{name}.npy"), act_np)

    if save_plots:
        np.save(os.path.join(out_dir, "encoder_output.npy"), out_np)

    grabber.remove()

    return activations_np, out_np, imgs


#%% ## Notebook example
# Paste the cell below into a notebook or run as a cell in an editor that
# supports interactive execution (for example VS Code with Python).

MODEL_ID = "5a183b56245743379d23efb485da8f4e"
IMAGE_IDS = [3387]
OUT_DIR = None  # or "/tmp/activations"

activations, encoder_out, imgs = run_inspect(
    model_id=MODEL_ID,
    image_ids=IMAGE_IDS,
    output_dir=OUT_DIR,
    max_layers=12,  # Increased to capture linear layers too
    save_plots=False,  # No files saved
)

# %%
# Check which layers are linear vs convolutional
print("\nLayer types and shapes:")
for i, name in enumerate(activations.keys()):
    act = activations[name]
    shape = act.shape
    
    # Determine layer type based on shape
    if act.ndim == 2 or (act.ndim == 3 and shape[1:] == (1, 1)):
        layer_type = "LINEAR"
    elif act.ndim == 4 or (act.ndim == 3 and len(shape) == 3):
        layer_type = "CONV/SPATIAL"
    else:
        layer_type = "UNKNOWN"
    
    print(f"  {i}: {name:30s} - shape: {str(shape):20s} - {layer_type}")

# %%
# Configurable layer selection - CHANGE THIS TO VIEW DIFFERENT LAYERS
LAYER_INDEX = 11  # 0 = first layer, 1 = second layer, etc. (max index is num_layers - 1)

# Get the selected layer
layer_names = list(activations.keys())
selected_layer = layer_names[LAYER_INDEX]
act = activations[selected_layer]

print(f"\nDisplaying layer: {selected_layer}")
print(f"Activation shape: {act.shape}")

# Check if this is a linear layer by looking at the shape
is_linear = act.ndim == 2 or (act.ndim == 3 and act.shape[1:] == (1, 1))

if is_linear or act.ndim == 2:
    # Linear layer activations: (N, Features) or (Features,)
    if act.ndim == 3:
        act = act.squeeze()  # Remove singleton dimensions
    if act.ndim == 2:
        act = act[0]  # Take first sample
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Activation values as bar plot
    axes[0, 0].bar(range(len(act)), act)
    axes[0, 0].set_title(f'{selected_layer} - All {len(act)} Activations')
    axes[0, 0].set_xlabel('Neuron Index')
    axes[0, 0].set_ylabel('Activation Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Activation distribution
    axes[0, 1].hist(act, bins=50, edgecolor='black')
    axes[0, 1].set_title('Activation Distribution')
    axes[0, 1].set_xlabel('Activation Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(0, color='red', linestyle='--', label='Zero')
    axes[0, 1].legend()
    
    # 3. Top-K active neurons
    k = min(20, len(act))
    top_k_indices = np.argsort(np.abs(act))[-k:][::-1]
    top_k_values = act[top_k_indices]
    axes[1, 0].barh(range(k), top_k_values)
    axes[1, 0].set_yticks(range(k))
    axes[1, 0].set_yticklabels([f'Neuron {i}' for i in top_k_indices])
    axes[1, 0].set_title(f'Top-{k} Active Neurons (by magnitude)')
    axes[1, 0].set_xlabel('Activation Value')
    axes[1, 0].invert_yaxis()
    
    # 4. Cumulative activation curve
    sorted_act = np.sort(np.abs(act))[::-1]
    cumsum = np.cumsum(sorted_act)
    cumsum_norm = cumsum / cumsum[-1] * 100
    axes[1, 1].plot(cumsum_norm)
    axes[1, 1].set_title('Cumulative Activation Energy')
    axes[1, 1].set_xlabel('Number of Neurons')
    axes[1, 1].set_ylabel('% of Total Activation')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(90, color='red', linestyle='--', label='90%')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Mean: {np.mean(act):.4f}")
    print(f"  Std: {np.std(act):.4f}")
    print(f"  Min: {np.min(act):.4f}")
    print(f"  Max: {np.max(act):.4f}")
    print(f"  % Zero/Near-zero (< 0.01): {np.sum(np.abs(act) < 0.01) / len(act) * 100:.1f}%")
    print(f"  % Negative: {np.sum(act < 0) / len(act) * 100:.1f}%")

elif act.ndim == 4:
    # Convolutional layer - take first sample
    act = act[0]

if act.ndim == 3:
    # (C, H, W) - show all channels
    C, H, W = act.shape
    n_show = C  # Show all channels
    ncols = min(8, C)  # Max 8 columns
    nrows = (n_show + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)
    
    for i in range(n_show):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        im = ax.imshow(act[i], cmap='viridis')
        ax.set_title(f'Filter {i}', fontsize=10)
        ax.axis('off')
    
    # Hide extra subplots
    for i in range(n_show, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].axis('off')
    
    plt.suptitle(f'{selected_layer} - All {C} filters', fontsize=14)
    plt.tight_layout()
    plt.show()

# %%
# Check what happens after layer 14
print("\nFinal encoder output stats:")
print(f"  Shape: {encoder_out.shape}")
print(f"  Mean: {encoder_out.mean():.4f}")
print(f"  Std: {encoder_out.std():.4f}")
print(f"  Min: {encoder_out.min():.4f}")
print(f"  Max: {encoder_out.max():.4f}")
print(f"  % in [0,100]: {((encoder_out >= 0) & (encoder_out <= 100)).mean() * 100:.1f}%")

# %%
# ## Analyze neuron activation across multiple images
# Run the inspection with multiple images to compare activation patterns

# CONFIGURATION
TOTAL_IMAGES = 8000  # Total number of images in dataset
BATCH_SIZE = 50  # Process images in batches to save memory (reduced to 50 for safety)
ACTIVATION_THRESHOLD = 0.01  # Threshold to consider a neuron "activated"

print(f"Processing {TOTAL_IMAGES} images in batches of {BATCH_SIZE}...")
print(f"Note: Reduce BATCH_SIZE if you run out of memory")

# Collect all activation statistics across batches
all_activation_stats = []

# Process images in batches
for batch_start in range(0, TOTAL_IMAGES, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, TOTAL_IMAGES)
    IMAGE_IDS_BATCH = list(range(batch_start, batch_end))
    
    print(f"\nProcessing batch: images {batch_start} to {batch_end-1}...")
    
    try:
        activations_batch, encoder_out_batch, imgs_batch = run_inspect(
            model_id=MODEL_ID,
            image_ids=IMAGE_IDS_BATCH,
            output_dir=None,
            max_layers=12,
            save_plots=False,
        )
        
        # Get the last layer activations
        last_layer_name = list(activations_batch.keys())[-1]
        last_layer_acts = activations_batch[last_layer_name]
        
        # Ensure we're working with (N, Features) shape
        if last_layer_acts.ndim == 3:
            last_layer_acts = last_layer_acts.squeeze()
        if last_layer_acts.ndim == 1:
            last_layer_acts = last_layer_acts[None, :]
        
        N_images_batch, N_features = last_layer_acts.shape
        
        # Calculate mode and statistics for each image in batch
        from scipy import stats
        
        for i in range(N_images_batch):
            act = last_layer_acts[i]
            
            # Calculate mode (most frequent value)
            # Round to avoid floating point precision issues
            act_rounded = np.round(act, decimals=4)
            mode_result = stats.mode(act_rounded, keepdims=True)
            mode_value = float(mode_result.mode[0])
            mode_count = int(mode_result.count[0])
            
            # Calculate other metrics
            n_activated = np.sum(np.abs(act) > ACTIVATION_THRESHOLD)
            pct_activated = (n_activated / N_features) * 100
            
            all_activation_stats.append({
                'image_id': IMAGE_IDS_BATCH[i],
                'mode_value': mode_value,
                'mode_count': mode_count,
                'mode_pct': (mode_count / N_features) * 100,
                'n_activated': n_activated,
                'pct_activated': pct_activated,
                'mean_abs_activation': np.mean(np.abs(act)),
                'max_activation': np.max(np.abs(act)),
            })
        
        print(f"  Processed {N_images_batch} images from this batch")
        
        # Clean up memory after each batch
        del activations_batch, encoder_out_batch, imgs_batch, last_layer_acts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"  Error processing batch {batch_start}-{batch_end}: {e}")
        # Clean up even on error
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        continue

print(f"\n{'='*80}")
print(f"Finished processing all batches. Total images processed: {len(all_activation_stats)}")
print(f"{'='*80}")

# Filter out images where mode is 0 or negative (we want only positive modes)
filtered_stats = [s for s in all_activation_stats if s['mode_value'] > 0.0]
print(f"\nFiltered out {len(all_activation_stats) - len(filtered_stats)} images with mode <= 0")
print(f"Kept {len(filtered_stats)} images with mode > 0")

# Sort by mode value (descending) - images with higher mode are more densely activated
sorted_by_activation = sorted(filtered_stats, key=lambda x: x['mode_value'], reverse=True)

# Display results
print(f"\n{'='*100}")
print(f"NEURON ACTIVATION ANALYSIS - Sorted by Mode (excluding images with mode <= 0)")
print(f"{'='*100}")
print(f"\n{'Rank':<6} {'Image ID':<10} {'Mode':<12} {'Mode Count':<12} {'Mode %':<12} {'# Act':<12} {'% Act':<12} {'Mean |Act|':<15} {'Max |Act|'}")
print(f"{'-'*100}")

for rank, info in enumerate(sorted_by_activation, 1):
    print(f"{rank:<6} {info['image_id']:<10} {info['mode_value']:<12.4f} "
          f"{info['mode_count']:<12} {info['mode_pct']:<12.1f} "
          f"{info['n_activated']:<12} {info['pct_activated']:<12.1f} "
          f"{info['mean_abs_activation']:<15.4f} {info['max_activation']:<15.4f}")

# Summary statistics
mode_values = [x['mode_value'] for x in filtered_stats]
n_activated_list = [x['n_activated'] for x in all_activation_stats]
print(f"\n{'='*100}")
print(f"SUMMARY STATISTICS (filtered: {len(filtered_stats)} images, excluded: {len(all_activation_stats) - len(filtered_stats)})")
print(f"{'='*100}")
print(f"Mode statistics (positive modes only):")
print(f"  Mean mode value: {np.mean(mode_values):.4f}")
print(f"  Std mode value: {np.std(mode_values):.4f}")
print(f"  Min mode value: {np.min(mode_values):.4f}")
print(f"  Max mode value: {np.max(mode_values):.4f}")
print(f"  Median mode value: {np.median(mode_values):.4f}")
print(f"\nActivation count statistics (all images):")
print(f"  Mean activated neurons: {np.mean(n_activated_list):.2f}")
print(f"  Std activated neurons: {np.std(n_activated_list):.2f}")
print(f"  Min activated neurons: {np.min(n_activated_list)}")
print(f"  Max activated neurons: {np.max(n_activated_list)}")
print(f"  Median activated neurons: {np.median(n_activated_list):.2f}")

# Visualization: Bar plot of mode values sorted by image
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Sorted by mode value
image_indices_sorted = [x['image_id'] for x in sorted_by_activation]
mode_values_sorted = [x['mode_value'] for x in sorted_by_activation]
axes[0].bar(range(len(image_indices_sorted)), mode_values_sorted, 
            color='steelblue', edgecolor='black')
axes[0].set_xlabel('Rank (sorted by mode value)')
axes[0].set_ylabel('Mode of Activation Values')
axes[0].set_title(f'Images Sorted by Mode of Neuron Activations (excluding mode <= 0)')
axes[0].axhline(np.mean(mode_values_sorted), color='red', linestyle='--', 
                label=f'Mean={np.mean(mode_values_sorted):.4f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Add image IDs as labels on x-axis (every nth to avoid crowding)
step = max(1, len(image_indices_sorted) // 20)
axes[0].set_xticks(range(0, len(image_indices_sorted), step))
axes[0].set_xticklabels([f"ID:{image_indices_sorted[i]}" for i in range(0, len(image_indices_sorted), step)], 
                        rotation=45, ha='right')

# Plot 2: Distribution of mode values
axes[1].hist(mode_values_sorted, bins=min(30, len(set(mode_values_sorted))), 
             color='coral', edgecolor='black', alpha=0.7)
axes[1].axvline(np.mean(mode_values_sorted), color='red', linestyle='--', 
                label=f'Mean={np.mean(mode_values_sorted):.4f}')
axes[1].axvline(np.median(mode_values_sorted), color='green', linestyle='--', 
                label=f'Median={np.median(mode_values_sorted):.4f}')
axes[1].set_xlabel('Mode of Activation Values')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Mode Values Across Images (positive modes only)')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# %%
print(f"\nTop 20 images with highest mode values (densest activation patterns):")
for rank, info in enumerate(sorted_by_activation[:20], 1):
    print(f"  {rank}. Image ID {info['image_id']}: mode={info['mode_value']:.4f} "
          f"({info['mode_count']} neurons = {info['mode_pct']:.1f}%)")

if len(sorted_by_activation) >= 20:
    print(f"\nBottom 20 images with lowest mode values (but mode > 0):")
    for rank, info in enumerate(sorted_by_activation[-20:], 1):
        print(f"  {rank}. Image ID {info['image_id']}: mode={info['mode_value']:.4f} "
              f"({info['mode_count']} neurons = {info['mode_pct']:.1f}%)")

print(f"\nImages excluded (mode <= 0):")
excluded = [s for s in all_activation_stats if s['mode_value'] <= 0.0]
if excluded:
    for info in excluded:
        print(f"  Image ID {info['image_id']}: mode={info['mode_value']:.4f}, "
              f"{info['n_activated']} neurons activated ({info['pct_activated']:.1f}%)")
else:
    print("  None - all images have mode > 0")

# %%
