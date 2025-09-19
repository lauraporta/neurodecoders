"""
Input optimization via gradient descent using a trained encoder.

Reconstruct an image from target firing rates by optimizing the input
image so that the encoder's predicted rates match the target under a
Poisson loss. The encoder is used in eval mode.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

# Local imports
from neurodecoders.extract_mei.mei import load_encoder_model
from neurodecoders.paths import get_path


def _to_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _total_variation(img: torch.Tensor) -> torch.Tensor:
    """Isotropic total variation regularizer for smoothness.

    Args:
        img: Tensor of shape (B, C, H, W)
    """
    dh = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
    dw = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
    return dh + dw


def create_comparison_plots(
    original_images: List[np.ndarray],
    reconstructed_images: List[np.ndarray],
    image_ids: List[int],
    output_path: str,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """Create comprehensive comparison plots with original,
    reconstructed, and difference images.

    Args:
        original_images: List of original images as numpy arrays
        reconstructed_images: List of reconstructed images as numpy arrays
        image_ids: List of image IDs for labeling
        output_path: Path to save the comparison plot
        figsize: Figure size tuple (width, height)
    """
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

    # Create subplots: 3 rows
    # (original, reconstructed, difference) x n_images cols
    fig, axes = plt.subplots(3, n_images, figsize=figsize)
    if n_images == 1:
        axes = axes.reshape(3, 1)

    for i, (orig, recon, img_id) in enumerate(
        zip(original_images, reconstructed_images, image_ids)
    ):
        # Ensure images are in [0, 1] range
        orig = np.clip(orig, 0, 1)
        recon = np.clip(recon, 0, 1)

        # Resize original image to match reconstructed image dimensions
        if orig.shape != recon.shape:
            print(f"[DEBUG] Resizing original {orig.shape} to match reconstructed {recon.shape}")
            try:
                # Try PIL first (most common and reliable)
                from PIL import Image

                # Convert to PIL Image - ensure we have a 2D image
                if len(orig.shape) == 3:
                    # If 3D, take the first channel or squeeze
                    if orig.shape[0] == 1:
                        orig_2d = orig.squeeze(0)
                    else:
                        orig_2d = orig[0]
                else:
                    orig_2d = orig

                pil_img = Image.fromarray(
                    (orig_2d * 255).astype(np.uint8)
                )

                # Resize to match reconstructed image
                if len(recon.shape) == 3:
                    target_size = (
                        recon.shape[2],  # width
                        recon.shape[1],   # height
                    )  # PIL uses (width, height)
                else:
                    target_size = (
                        recon.shape[1],  # width
                        recon.shape[0],  # height
                    )  # PIL uses (width, height)
                pil_img = pil_img.resize(target_size, Image.LANCZOS)

                # Convert back to numpy
                orig = np.array(pil_img) / 255.0
                # Ensure the shape matches the reconstructed image
                if len(recon.shape) == 3 and len(orig.shape) == 2:
                    orig = orig.reshape(1, orig.shape[0], orig.shape[1])
                elif len(recon.shape) == 2 and len(orig.shape) == 3:
                    orig = orig.squeeze()
                # Ensure we have a 2D image for display
                if len(orig.shape) == 1:
                    # If we somehow got a 1D array, reshape it to 2D
                    orig = orig.reshape(int(np.sqrt(len(orig))), int(np.sqrt(len(orig))))
                print(f"[DEBUG] PIL resized original image to {orig.shape}")
            except ImportError:
                try:
                    # Try scipy as fallback
                    from scipy.ndimage import zoom

                    zoom_factors = [
                        recon.shape[j] / orig.shape[j]
                        for j in range(len(orig.shape))
                    ]
                    orig = zoom(orig, zoom_factors, order=1)
                    # Ensure the shape matches the reconstructed image
                    if len(recon.shape) == 3 and len(orig.shape) == 2:
                        orig = orig.reshape(1, orig.shape[0], orig.shape[1])
                    elif len(recon.shape) == 2 and len(orig.shape) == 3:
                        orig = orig.squeeze()
                    # Ensure we have a 2D image for display
                    if len(orig.shape) == 1:
                        orig = orig.reshape(int(np.sqrt(len(orig))), int(np.sqrt(len(orig))))
                    print(
                        f"[DEBUG] Scipy resized original image to {orig.shape}"
                    )
                except ImportError:
                    # Final fallback: simple numpy resizing using interpolation
                    # Create coordinate arrays for interpolation
                    orig_h, orig_w = orig.shape[:2]
                    recon_h, recon_w = recon.shape[:2]

                    # Create coordinate grids
                    y_coords = np.linspace(0, orig_h - 1, recon_h)
                    x_coords = np.linspace(0, orig_w - 1, recon_w)

                    # Simple nearest neighbor interpolation
                    y_indices = np.round(y_coords).astype(int)
                    x_indices = np.round(x_coords).astype(int)

                    # Ensure indices are within bounds
                    y_indices = np.clip(y_indices, 0, orig_h - 1)
                    x_indices = np.clip(x_indices, 0, orig_w - 1)

                    # Resize using advanced indexing
                    if len(orig.shape) == 3:
                        orig = orig[y_indices[:, None], x_indices[None, :]]
                    else:
                        orig = orig[y_indices[:, None], x_indices[None, :]]
                    
                    # Ensure the shape matches the reconstructed image
                    if len(recon.shape) == 3 and len(orig.shape) == 2:
                        orig = orig.reshape(1, orig.shape[0], orig.shape[1])
                    elif len(recon.shape) == 2 and len(orig.shape) == 3:
                        orig = orig.squeeze()
                    # Ensure we have a 2D image for display
                    if len(orig.shape) == 1:
                        orig = orig.reshape(int(np.sqrt(len(orig))), int(np.sqrt(len(orig))))

        # Calculate difference
        diff = np.abs(orig - recon)

        # Original image
        axes[0, i].imshow(orig.squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Original {img_id}", fontsize=12)
        axes[0, i].axis("off")

        # Reconstructed image
        axes[1, i].imshow(recon.squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"Reconstructed {img_id}", fontsize=12)
        axes[1, i].axis("off")

        # Difference image
        im = axes[2, i].imshow(diff.squeeze(), cmap="hot", vmin=0, vmax=1)
        axes[2, i].set_title(f"Difference {img_id}", fontsize=12)
        axes[2, i].axis("off")

    # Add a single colorbar for all difference images
    if n_images > 0:
        # Create a colorbar for the difference images
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Difference', rotation=270, labelpad=15)
        # Don't use tight_layout when we have custom colorbar
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    
    plt.close()


@dataclass
class OptimConfig:
    image_size: int = 64
    channels: int = 1
    steps: int = 2000
    lr: float = 0.05
    tv_weight: float = 1e-4
    l2_weight: float = 1e-6
    log_every: int = 50
    init_mean: float = 0.5
    init_std: float = 0.01
    clamp_min: float = 0.0
    clamp_max: float = 1.0
    seed: Optional[int] = 42
    image_ids: Optional[list] = None


class ImageOptimizer:
    """Optimize an input image to match target firing rates.

    Uses Poisson NLL loss between predicted rates and target rates. Applies
    softplus to encoder outputs to ensure positive rates.
    """

    def __init__(
        self,
        encoder: nn.Module,
        target_rates: np.ndarray,
        config: OptimConfig,
    ) -> None:
        self.device = _to_device()
        self.encoder = encoder.to(self.device)
        self.encoder.eval()

        if target_rates is None:
            raise ValueError("target_rates is required and cannot be None")

        self.target = torch.tensor(target_rates.astype(np.float32)).to(
            self.device
        )
        if self.target.ndim != 1:
            raise ValueError("target_rates must be a 1D array of shape (N,)")

        self.cfg = config
        if self.cfg.channels != 1:
            raise ValueError(
                "Only grayscale images are supported. Set channels=1."
            )

        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        # Initialize from a gray image (all zeros in [0, 1] range)
        init = torch.zeros(
            1, self.cfg.channels, self.cfg.image_size, self.cfg.image_size
        )
        init = init.clamp(self.cfg.clamp_min, self.cfg.clamp_max)
        self.image = nn.Parameter(init.to(self.device))

        # Validate encoder compatibility (channels/size and output dims)
        try:
            with torch.no_grad():
                test_out = self.encoder(self.image)
            if test_out.ndim == 2 and test_out.shape[0] == 1:
                test_out = test_out[0]
            if test_out.ndim != 1:
                raise ValueError("Encoder output must be shape (1, N) or (N,)")
            if test_out.shape[0] != self.target.shape[0]:
                raise ValueError(
                    "Target length does not match encoder output: "
                    f"got {test_out.shape[0]}, "
                    f"expected {self.target.shape[0]}"
                )
        except Exception as e:
            raise ValueError(
                "Encoder is incompatible with the provided image shape."
                f" Image shape: (1, {self.cfg.channels}, "
                f"{self.cfg.image_size},"
                f" {self.cfg.image_size}). Original error: {e}"
            ) from e

        self.poisson = nn.PoissonNLLLoss(log_input=False, reduction="mean")
        self.softplus = nn.Softplus()

    def step(
        self, opt: optim.Optimizer
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        opt.zero_grad(set_to_none=True)

        pred = self.encoder(self.image)
        if pred.ndim == 2 and pred.shape[0] == 1:
            pred = pred[0]
        elif pred.ndim != 1:
            raise ValueError("Encoder output must be shape (1, N) or (N,)")

        rate = self.softplus(pred)
        loss_data = self.poisson(rate, self.target)

        tv = _total_variation(self.image) if self.cfg.tv_weight > 0 else 0.0
        l2 = (self.image**2).mean() if self.cfg.l2_weight > 0 else 0.0

        total = (
            loss_data
            + self.cfg.tv_weight * (tv if isinstance(tv, torch.Tensor) else 0)
            + self.cfg.l2_weight * (l2 if isinstance(l2, torch.Tensor) else 0)
        )

        total.backward()
        opt.step()
        with torch.no_grad():
            self.image.clamp_(self.cfg.clamp_min, self.cfg.clamp_max)

        metrics = {
            "loss_total": float(total.detach().cpu().item()),
            "loss_poisson": float(loss_data.detach().cpu().item()),
            "tv": float(tv.detach().cpu().item())
            if isinstance(tv, torch.Tensor)
            else 0.0,
            "l2": float(l2.detach().cpu().item())
            if isinstance(l2, torch.Tensor)
            else 0.0,
        }
        return self.image.detach(), metrics

    def optimize(self) -> Tuple[np.ndarray, Dict[str, float]]:
        opt = optim.Adam([self.image], lr=self.cfg.lr)
        last_metrics: Dict[str, float] = {}

        for step in range(1, self.cfg.steps + 1):
            img, metrics = self.step(opt)
            last_metrics = metrics
            if step % self.cfg.log_every == 0:
                try:
                    mlflow.log_metrics(metrics, step=step)
                except Exception:
                    pass

        img_np = img.squeeze(0).detach().cpu().numpy()
        return img_np, last_metrics

    def save_image(self, path: str) -> None:
        with torch.no_grad():
            save_image(self.image.clamp(0, 1), path)

    def reconstruct_multiple_images(
        self,
        target_rates_list: List[np.ndarray],
        image_ids: List[int],
        output_dir: str,
        original_images: Optional[List[np.ndarray]] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Reconstruct multiple images and create comparison plots.

        Args:
            target_rates_list: List of target firing rate arrays
            image_ids: List of image IDs for labeling
            output_dir: Directory to save outputs
            original_images: Optional list of original images for comparison

        Returns:
            Tuple of (original_images, reconstructed_images) lists
        """
        reconstructed_images = []

        for i, (target_rates, img_id) in enumerate(
            zip(target_rates_list, image_ids)
        ):
            # Create new optimizer instance for each image
            optimizer = ImageOptimizer(
                encoder=self.encoder,
                target_rates=target_rates,
                config=self.cfg,
            )

            # Optimize the image
            reconstructed_img, metrics = optimizer.optimize()
            reconstructed_images.append(reconstructed_img)

            # Save individual reconstruction
            individual_path = os.path.join(
                output_dir, f"reconstruction_{img_id}.png"
            )
            optimizer.save_image(individual_path)

            # Save numpy array
            npy_path = os.path.join(output_dir, f"reconstruction_{img_id}.npy")
            np.save(npy_path, reconstructed_img)

            print(
                f"Reconstructed image {img_id} - "
                f"Loss: {metrics.get('loss_total', 'N/A'):.4f}"
            )

        # Create comparison plot if we have multiple images
        if len(reconstructed_images) > 1:
            comparison_path = os.path.join(output_dir, "comparison_plot.png")
            create_comparison_plots(
                original_images=original_images
                or [np.zeros_like(img) for img in reconstructed_images],
                reconstructed_images=reconstructed_images,
                image_ids=image_ids,
                output_path=comparison_path,
            )
            print(f"Comparison plot saved: {comparison_path}")
        elif len(reconstructed_images) == 1:
            # Even for single image,
            # create a comparison plot if we have original
            if original_images and len(original_images) == 1:
                comparison_path = os.path.join(
                    output_dir, "comparison_plot.png"
                )
                create_comparison_plots(
                    original_images=original_images,
                    reconstructed_images=reconstructed_images,
                    image_ids=image_ids,
                    output_path=comparison_path,
                )
                print(f"Single image comparison plot saved: {comparison_path}")

        return original_images or [
            np.zeros_like(img) for img in reconstructed_images
        ], reconstructed_images


def load_encoder_from_mlflow(model_id: str) -> nn.Module:
    """Load a PyTorch encoder from MLflow given a model identifier.

    Tries common registry URIs; falls back to latest run's encoder_model.
    Also handles local file paths.
    """
    # Check if it's a local file path first
    if os.path.exists(model_id):
        try:
            device = _to_device()
            return load_encoder_model(model_id, device)
        except Exception as e:
            print(f"Warning: Could not load local model {model_id}: {e}")
    
    tried = []
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

    # Fallback: attempt to load the most recent run's encoder_model artifact
    try:
        from mlflow.tracking import MlflowClient

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
                for art_name in ("encoder_model", "model"):
                    try:
                        uri = f"runs:/{run_id}/{art_name}"
                        model = mlflow.pytorch.load_model(uri)
                        return model
                    except Exception:
                        continue
    except Exception:
        pass

    # Local fallback: load most recent encoder weights from workspace
    try:
        models_dir = get_path("workspace/models/encoders")
        if os.path.exists(models_dir):
            cand = [
                os.path.join(models_dir, f)
                for f in os.listdir(models_dir)
                if f.endswith(".pth") or f.endswith(".ckpt")
            ]
            if cand:
                latest = max(cand, key=os.path.getmtime)
                device = _to_device()
                return load_encoder_model(latest, device)
    except Exception:
        pass

    raise RuntimeError(
        "Could not load encoder from MLflow. Tried URIs: " + ", ".join(tried)
    )
