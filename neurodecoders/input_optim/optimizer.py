"""
Input optimization via gradient descent using a trained encoder.

Reconstruct an image from target firing rates by optimizing the input
image so that the encoder's predicted rates match the target under a
Poisson loss. The encoder is used in eval mode.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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


@dataclass
class OptimConfig:
    image_size: int = 64
    channels: int = 1
    steps: int = 2000
    lr: float = 0.001
    tv_weight: float = 1e-4
    l2_weight: float = 1e-8
    log_every: int = 50
    init_mean: float = 0.5
    init_std: float = 0.1
    clamp_min: float = 0.0
    clamp_max: float = 1.0
    seed: Optional[int] = 42


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

        # Initialize from random noise in [0, 1] range
        init = (
            torch.randn(
                1, self.cfg.channels, self.cfg.image_size, self.cfg.image_size
            )
            * self.cfg.init_std
            + self.cfg.init_mean
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
        # L2 regularization around mean instead of zero to avoid black bias
        l2 = (
            ((self.image - self.cfg.init_mean) ** 2).mean()
            if self.cfg.l2_weight > 0
            else 0.0
        )

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
            "image_mean": float(self.image.mean().detach().cpu().item()),
            "image_std": float(self.image.std().detach().cpu().item()),
            "rate_mean": float(rate.mean().detach().cpu().item()),
            "target_mean": float(self.target.mean().detach().cpu().item()),
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


def load_encoder_from_mlflow(model_id: str) -> nn.Module:
    """Load a PyTorch encoder from MLflow given a model identifier.

    Tries common registry URIs; falls back to latest run's encoder_model.
    """
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
