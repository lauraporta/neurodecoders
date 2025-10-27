"""
Decoder image generation and visualization utilities.

This module provides functionality to load trained decoder models and generate
images from neural firing rates, with comprehensive plotting capabilities
similar to the input optimization plots.
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
from mlflow.tracking import MlflowClient

from neurodecoders.decoder.models import MirrorSimpleEncoderDecoder, SimpleDecoder
from neurodecoders.data.loading import load_npz_dataset, normalize_images_and_rates
from neurodecoders.paths import get_path


def load_decoder_from_mlflow(model_id: str) -> nn.Module:
    """Load a PyTorch decoder from MLflow given a model identifier.
    
    Args:
        model_id: MLflow model ID, run URI, or local file path
        
    Returns:
        Loaded decoder model
    """
    # Check if it's a local file path first
    if os.path.exists(model_id):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() 
                                else "mps" if torch.backends.mps.is_available() 
                                else "cpu")
            return load_decoder_model(model_id, device)
        except Exception as e:
            print(f"Warning: Could not load local model {model_id}: {e}")
    
    tried = []
    
    # Try direct run URI first (e.g., runs:/run_id/model)
    if model_id.startswith("runs:/"):
        try:
            model = mlflow.pytorch.load_model(model_id)
            return model
        except Exception as e:
            tried.append(model_id)
            print(f"Warning: Could not load run URI {model_id}: {e}")
    
    # Try model registry URIs
    uris = [
        f"models:/{model_id}",
        f"models:/{model_id}/latest", 
        f"models:/{model_id}/Production",
        f"models:/{model_id}/Staging",
    ]
    
    for uri in uris:
        try:
            model = mlflow.pytorch.load_model(uri)
            return model
        except Exception:
            tried.append(uri)
    
    # Fallback: attempt to load the most recent run's decoder_model artifact
    try:
        client = MlflowClient()
        exps = client.list_experiments()
        exp_ids = [e.experiment_id for e in exps]
        if exp_ids:
            runs = client.search_runs(
                experiment_ids=exp_ids,
                order_by=["attributes.start_time DESC"],
                max_results=50,
            )
            for r in runs:
                run_id = r.info.run_id
                for art_name in ("decoder_model", "model"):
                    try:
                        uri = f"runs:/{run_id}/{art_name}"
                        model = mlflow.pytorch.load_model(uri)
                        return model
                    except Exception:
                        continue
    except Exception:
        pass
    
    # Local fallback: load most recent decoder weights from workspace
    try:
        workspace_path = get_path("workspace/models")
        model_files = list(Path(workspace_path).glob("decoder_*.pth"))
        if model_files:
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            device = torch.device("cuda" if torch.cuda.is_available() 
                               else "mps" if torch.backends.mps.is_available() 
                               else "cpu")
            return load_decoder_model(str(latest_model), device)
    except Exception:
        pass
    
    raise ValueError(f"Could not load decoder model {model_id}. Tried: {tried}")


def load_decoder_model(model_path: str, device: torch.device) -> nn.Module:
    """Load a decoder model from a local file path.
    
    Args:
        model_path: Path to the model file
        device: Device to load the model on
        
    Returns:
        Loaded decoder model
    """
    # Try to infer model type from the model file
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if it's a Lightning checkpoint
    if "state_dict" in checkpoint:
        # Extract hyperparameters to determine model type
        hparams = checkpoint.get("hyper_parameters", {})
        in_neurons = hparams.get("in_neurons", 100)
        image_size = hparams.get("image_size", 64)
        model_type = hparams.get("model_type", "simple")
        
        if model_type == "mirror_simple":
            model = MirrorSimpleEncoderDecoder(in_neurons, image_size)
        else:
            model = SimpleDecoder(in_neurons, image_size)
        
        # Load the state dict
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model
    else:
        # Assume it's a direct model state dict
        # Handle the case where state dict has "model." prefix
        state_dict = checkpoint
        if any(key.startswith("model.") for key in state_dict.keys()):
            # Remove "model." prefix from all keys
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("model."):
                    new_key = key[6:]  # Remove "model." prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        # Infer architecture from state dict
        # Look for the first linear layer to get input neurons
        in_neurons = 100  # Default
        for key, value in state_dict.items():
            if "fc.0.weight" in key:
                in_neurons = value.shape[1]  # Input dimension
                break
        
        # Infer image size from the final output layer
        image_size = 64  # Default
        for key, value in state_dict.items():
            if "deconv.12.weight" in key:  # Final conv layer
                # This is a rough estimate - we'd need to trace through the architecture
                # For now, use a reasonable default
                image_size = 64
                break
        
        model = SimpleDecoder(in_neurons, image_size)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model


def load_test_data(dataset_path: str, n_images: int = 10) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Load test data and select random images.
    
    Args:
        dataset_path: Path to the test dataset
        n_images: Number of random images to select
        
    Returns:
        Tuple of (images, firing_rates, selected_indices)
    """
    images, firing_rates = load_npz_dataset(dataset_path)
    images, firing_rates, _, _ = normalize_images_and_rates(images, firing_rates)
    
    # Select random images
    n_total = len(images)
    n_images = min(n_images, n_total)
    selected_indices = random.sample(range(n_total), n_images)
    
    selected_images = images[selected_indices]
    selected_firing_rates = firing_rates[selected_indices]
    
    return selected_images, selected_firing_rates, selected_indices


def create_decoder_comparison_plots(
    original_images: List[np.ndarray],
    generated_images: List[np.ndarray], 
    image_ids: List[int],
    output_path: str,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """Create comprehensive comparison plots for decoder-generated images.
    
    Args:
        original_images: List of original images as numpy arrays
        generated_images: List of decoder-generated images as numpy arrays
        image_ids: List of image IDs for labeling
        output_path: Path to save the comparison plot
        figsize: Figure size tuple (width, height)
    """
    import matplotlib.pyplot as plt
    
    n_images = len(original_images)
    if n_images == 0:
        return
    
    # Adjust figure size based on number of images
    if n_images == 1:
        figsize = (6, 8)  # Taller for single image
    elif n_images <= 3:
        figsize = (4 * n_images, 8)  # 4 units per image
    else:
        figsize = (12, 8)  # Cap at reasonable size for many images
    
    # Create subplots: 3 rows (original, generated, difference) x n_images cols
    fig, axes = plt.subplots(3, n_images, figsize=figsize)
    if n_images == 1:
        axes = axes.reshape(3, 1)
    
    for i, (orig, gen, img_id) in enumerate(zip(original_images, generated_images, image_ids)):
        # Ensure images are in [0, 1] range
        orig = np.clip(orig, 0, 1)
        gen = np.clip(gen, 0, 1)
        
        # Resize original image to match generated image dimensions if needed
        if orig.shape != gen.shape:
            from PIL import Image
            if len(orig.shape) == 3:
                if orig.shape[0] == 1:
                    orig_2d = orig.squeeze(0)
                else:
                    orig_2d = orig[0]
            else:
                orig_2d = orig
                
            pil_img = Image.fromarray((orig_2d * 255).astype(np.uint8))
            target_size = (gen.shape[2], gen.shape[1]) if len(gen.shape) == 3 else (gen.shape[1], gen.shape[0])
            pil_img = pil_img.resize(target_size, Image.LANCZOS)
            orig = np.array(pil_img) / 255.0
            
            if len(gen.shape) == 3 and len(orig.shape) == 2:
                orig = orig.reshape(1, orig.shape[0], orig.shape[1])
            elif len(gen.shape) == 2 and len(orig.shape) == 3:
                orig = orig.squeeze()
        
        # Calculate difference
        diff = np.abs(orig - gen)
        
        # Plot original image
        if len(orig.shape) == 3:
            orig_display = orig.squeeze(0) if orig.shape[0] == 1 else orig[0]
        else:
            orig_display = orig
        axes[0, i].imshow(orig_display, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original {img_id}')
        axes[0, i].axis('off')
        
        # Plot generated image
        if len(gen.shape) == 3:
            gen_display = gen.squeeze(0) if gen.shape[0] == 1 else gen[0]
        else:
            gen_display = gen
        axes[1, i].imshow(gen_display, cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Generated {img_id}')
        axes[1, i].axis('off')
        
        # Plot difference
        if len(diff.shape) == 3:
            diff_display = diff.squeeze(0) if diff.shape[0] == 1 else diff[0]
        else:
            diff_display = diff
        im = axes[2, i].imshow(diff_display, cmap='hot', vmin=0, vmax=1)
        axes[2, i].set_title(f'Difference {img_id}')
        axes[2, i].axis('off')
    
    # Add row labels
    axes[0, 0].set_ylabel('Original', rotation=90, size='large')
    axes[1, 0].set_ylabel('Generated', rotation=90, size='large')
    axes[2, 0].set_ylabel('Difference', rotation=90, size='large')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Decoder comparison plot saved: {output_path}")


def generate_decoder_images(
    decoder: nn.Module,
    firing_rates: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Generate images from neural firing rates using the decoder.
    
    Args:
        decoder: Trained decoder model
        firing_rates: Neural firing rates array
        device: Device to run inference on
        
    Returns:
        Generated images as numpy array
    """
    decoder.eval()
    decoder.to(device)
    
    with torch.no_grad():
        firing_rates_tensor = torch.tensor(firing_rates, dtype=torch.float32).to(device)
        generated_images = decoder(firing_rates_tensor)
        generated_images = generated_images.cpu().numpy()
    
    # Ensure images are in [0, 1] range
    generated_images = np.clip(generated_images, 0, 1)
    
    return generated_images


def main():
    """Main function to generate decoder images and create comparison plots."""
    parser = argparse.ArgumentParser(description="Generate decoder images and create plots")
    parser.add_argument("--model_id", type=str, required=True,
                       help="MLflow model ID or path to decoder model")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to test dataset")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for plots")
    parser.add_argument("--n_images", type=int, default=10,
                       help="Number of random images to generate")
    parser.add_argument("--tracking_uri", type=str, default=None,
                       help="MLflow tracking URI")
    parser.add_argument("--run_name", type=str, default=None,
                       help="MLflow run name")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set up MLflow
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    else:
        # Use tracking URI from config (supports database)
        from neurodecoders.config import get_mlflow_tracking_uri
        tracking_uri = get_mlflow_tracking_uri()
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
    
    # Set up output directory
    out_dir = (args.output_dir if args.output_dir is not None 
               else get_path("workspace/plots/decoder_generation"))
    os.makedirs(out_dir, exist_ok=True)
    
    # Set up MLflow run
    run_name = (args.run_name if args.run_name is not None 
                else f"decoder_generation_{args.model_id}")
    
    with mlflow.start_run(run_name=run_name, log_system_metrics=True):
        # Log parameters
        mlflow.log_params({
            "model_id": args.model_id,
            "dataset_path": args.dataset_path,
            "n_images": args.n_images,
            "seed": args.seed,
        })
        
        # Load decoder model
        print(f"Loading decoder model: {args.model_id}")
        decoder = load_decoder_from_mlflow(args.model_id)
        
        # Load test data
        print(f"Loading test data from: {args.dataset_path}")
        images, firing_rates, selected_indices = load_test_data(
            args.dataset_path, args.n_images
        )
        
        # Generate images
        print(f"Generating images for {len(images)} samples...")
        device = torch.device("cuda" if torch.cuda.is_available() 
                            else "mps" if torch.backends.mps.is_available() 
                            else "cpu")
        generated_images = generate_decoder_images(decoder, firing_rates, device)
        
        # Create comparison plots
        print("Creating comparison plots...")
        comparison_path = os.path.join(out_dir, "decoder_comparison_plot.png")
        create_decoder_comparison_plots(
            original_images=list(images),
            generated_images=list(generated_images),
            image_ids=selected_indices,
            output_path=comparison_path,
        )
        
        # Log artifacts with proper organization
        try:
            # Log the comparison plot with a clear artifact path
            mlflow.log_artifact(comparison_path, "decoder_comparison_plot.png")
            print(f"✅ Artifact logged to MLflow: {comparison_path}")
        except Exception as e:
            print(f"⚠️  Could not log artifact to MLflow: {e}")
            print(f"   Plot saved locally: {comparison_path}")
        
        # Log metrics (handle shape mismatch by resizing)
        if images.shape != generated_images.shape:
            from PIL import Image
            import torch.nn.functional as F
            
            # Resize original images to match generated images
            resized_images = []
            for img in images:
                if len(img.shape) == 3:
                    img_2d = img.squeeze(0) if img.shape[0] == 1 else img[0]
                else:
                    img_2d = img
                
                pil_img = Image.fromarray((img_2d * 255).astype(np.uint8))
                target_size = (generated_images.shape[2], generated_images.shape[1]) if len(generated_images.shape) == 3 else (generated_images.shape[1], generated_images.shape[0])
                pil_img = pil_img.resize(target_size, Image.LANCZOS)
                resized_img = np.array(pil_img) / 255.0
                
                if len(generated_images.shape) == 3 and len(resized_img.shape) == 2:
                    resized_img = resized_img.reshape(1, resized_img.shape[0], resized_img.shape[1])
                elif len(generated_images.shape) == 2 and len(resized_img.shape) == 3:
                    resized_img = resized_img.squeeze()
                
                resized_images.append(resized_img)
            
            images_for_metrics = np.array(resized_images)
        else:
            images_for_metrics = images
        
        mse_loss = np.mean((images_for_metrics - generated_images) ** 2)
        mae_loss = np.mean(np.abs(images_for_metrics - generated_images))
        
        mlflow.log_metric("mse_loss", mse_loss)
        mlflow.log_metric("mae_loss", mae_loss)
        
        print(f"Generated {len(generated_images)} images")
        print(f"MSE Loss: {mse_loss:.4f}")
        print(f"MAE Loss: {mae_loss:.4f}")
        print(f"Comparison plot saved: {comparison_path}")


if __name__ == "__main__":
    main()
