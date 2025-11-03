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
    original_image = original_images[0]
    reconstructed_image = reconstructed_images[0]
    image_id = image_ids[0]

    figsize = (4, 2) 

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Calculate difference
    diff = original_image - reconstructed_image

    # Original image
    axes[0].imshow(original_image.squeeze(), cmap="gray")
    axes[0].set_title(f"Original {image_id}", fontsize=12)
    axes[0].axis("off")

    # Reconstructed image
    axes[1].imshow(reconstructed_image.squeeze(), cmap="gray")
    axes[1].set_title(f"Reconstructed {image_id}", fontsize=12)
    axes[1].axis("off")

    # Difference image
    im = axes[2].imshow(diff.squeeze(), cmap="hot")
    axes[2].set_title(f"Difference {image_id}", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    
    plt.close()


@dataclass
class OptimConfig:
    image_size: int = 64
    channels: int = 1
    steps: int = 2000
    lr: float = 0.05
    log_every: int = 20
    init_mean: float = 0.5
    init_std: float = 0.01
    clamp_min: float = -1
    clamp_max: float = 1
    seed: Optional[int] = 42
    image_ids: Optional[list] = None
    loss: str = "poisson_mean"


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

        target_variance = torch.var(self.target).item()
        self.idx_of_top_30_by_variance = torch.topk(
            torch.abs(self.target - torch.mean(self.target)), k=30
        ).indices.tolist()

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

        # Loss function
        if self.cfg.loss == "mse":
            self.loss = nn.MSELoss(reduction="mean")
        elif self.cfg.loss == "poisson_mean":
            self.loss = nn.PoissonNLLLoss(log_input=False, reduction="mean")
        elif self.cfg.loss == "poisson_sum":
            self.loss = nn.PoissonNLLLoss(log_input=False, reduction="sum")
        else:
            raise ValueError(f"Unknown loss function: {self.cfg.loss}")
        self.softplus = nn.Softplus()

    def step(
        self, opt: optim.Optimizer
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        opt.zero_grad(set_to_none=True)

        def apply_gaussian_blur_to_image(img: torch.Tensor) -> torch.Tensor:
            """Applies a Gaussian blur to the input image tensor."""
            import torch.nn.functional as F

            # Define a simple Gaussian kernel
            kernel_size = 20
            sigma = 5.0
            x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
            x_grid = x.repeat(kernel_size).view(kernel_size, kernel_size)
            y_grid = x_grid.t()
            gaussian_kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
            gaussian_kernel /= gaussian_kernel.sum()

            # Reshape to [out_channels, in_channels, kH, kW]
            gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).to(img.device)

            # Apply the Gaussian filter
            img = F.conv2d(img, gaussian_kernel, padding=kernel_size // 2)
            return img

        pred = self.encoder(apply_gaussian_blur_to_image(self.image))
        if pred.ndim == 2 and pred.shape[0] == 1:
            pred = pred[0]
        elif pred.ndim != 1:
            raise ValueError("Encoder output must be shape (1, N) or (N,)")

        rate = self.softplus(pred)

        # only use the idx_of_top_30_by_variance
        loss = self.loss(
            rate[self.idx_of_top_30_by_variance], 
            self.target[self.idx_of_top_30_by_variance])

        loss.backward()

        with torch.no_grad():
            if self.image.grad is not None:
                # Normalize by matrix norm (Frobenius norm)
                grad_norm = torch.norm(self.image.grad)
                if grad_norm > 0:
                    self.image.grad /= grad_norm
                # Clip gradients to [-1, 1]
                self.image.grad.clamp_(-1.0, 1.0)

        opt.step()
        with torch.no_grad():
            # #  normalise first
            # self.image.data -= self.image.data.mean()
            # self.image.data /= self.image.data.std()
            self.image.data.clamp_(self.cfg.clamp_min, self.cfg.clamp_max)

        metrics = {
            "loss": float(loss.detach().cpu().item()),
        }
        return self.image.detach(), metrics

    def optimize(self) -> Tuple[np.ndarray, Dict[str, float]]:
        opt = optim.Adam([self.image], lr=self.cfg.lr)
        # Add CosineAnnealingLR scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cfg.steps, eta_min=1e-6
        )
        last_metrics: Dict[str, float] = {}

        for step in range(1, self.cfg.steps + 1):
            img, metrics = self.step(opt)
            last_metrics = metrics
            # Step the scheduler with current loss
            scheduler.step(metrics["loss"])
            if step % self.cfg.log_every == 0:
                try:
                    mlflow.log_metrics(metrics, step=step)
                except Exception:
                    pass

        img_np = img.squeeze(0).detach().cpu().numpy()
        return img_np, last_metrics

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
        return original_images or [
            np.zeros_like(img) for img in reconstructed_images
        ], reconstructed_images


def load_encoder_from_mlflow(model_id: str) -> nn.Module:
    """Load a PyTorch encoder from MLflow given a model identifier.

    Tries common registry URIs; falls back to latest run's encoder_model.
    Also handles local file paths and run IDs.
    
    Args:
        model_id: Can be a model registry ID (m-xxx), run ID (32 char hex), or file path
    """
    tried = []
    
    # Check if it's a run ID (32 char hex string without m- prefix)
    if len(model_id) == 32 and not model_id.startswith('m-'):
        print(f"[INFO] Input looks like a run ID, trying to load from run artifacts: {model_id}")
        uri = f"runs:/{model_id}/model"
        try:
            print(f"[INFO] Trying to load from: {uri}")
            model = mlflow.pytorch.load_model(uri)
            print(f"[INFO] Successfully loaded model from {uri}")
            return model
        except Exception as e:
            print(f"[WARNING] Failed to load from {uri}: {e}")
            tried.append(uri)

    raise RuntimeError(
        "Could not load encoder from MLflow. Tried URIs: " + ", ".join(tried)
    )
