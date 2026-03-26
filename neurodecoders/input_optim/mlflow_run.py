#!/usr/bin/env python3
"""
Run input optimization via gradient descent with MLflow tracking.

Loads a trained encoder from MLflow by model ID and optimizes an input
image so that the encoder's predicted firing rates match provided target
rates under a Poisson loss.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List

import mlflow
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim

from neurodecoders.data.loading import (
    compute_normalization_stats, 
    load_npz_dataset
)
from neurodecoders.input_optim.optimizer import (
    ImageOptimizer,
    OptimConfig,
    load_encoder_from_mlflow,
)
from neurodecoders.mlflow_utils.utils import (
    log_single_artifact,
    setup_mlflow_experiment,
)
from neurodecoders.paths import get_path
from mlflow.tracking import MlflowClient


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
client = MlflowClient(MLFLOW_TRACKING_URI) 


@dataclass
class DatasetInfo:
    """Encapsulates all dataset information and data for a model.
    
    This class is created once per model and provides access to:
    - Model parameters (from MLflow)
    - Dataset metadata (filename, path, type)
    - Dataset arrays (images and firing rates)
    - Computed properties (image size)
    """
    model_id: str
    params: dict
    filename: str
    path: str
    dataset_type: str
    images: np.ndarray
    firing_rates: np.ndarray
    
    @property
    def image_size(self) -> int:
        """Get image size (assumes square images).
        
        Returns:
            Image size (width/height)
            
        Raises:
            ValueError: If images are not square or have unexpected dimensions
        """
        if self.images.ndim == 4:
            # (N, C, H, W)
            if self.images.shape[2] != self.images.shape[3]:
                raise ValueError(
                    f"Non-square images not supported. Got shape {self.images.shape}"
                )
            return self.images.shape[2]
        elif self.images.ndim == 3:
            # (N, H, W)
            if self.images.shape[1] != self.images.shape[2]:
                raise ValueError(
                    f"Non-square images not supported. Got shape {self.images.shape}"
                )
            return self.images.shape[1]
        else:
            raise ValueError(
                f"Unexpected image array shape: {self.images.shape}. "
                "Expected 3D (N, H, W) or 4D (N, C, H, W)"
            )
    
    def get_images(self, image_ids: List[int]) -> List[np.ndarray]:
        """Get specific images by their indices.
        
        Args:
            image_ids: List of image indices to retrieve
            
        Returns:
            List of image arrays
            
        Raises:
            IndexError: If any image_id is out of range
        """
        result = []
        for img_id in image_ids:
            if img_id < 0 or img_id >= self.images.shape[0]:
                raise IndexError(
                    f"image_id {img_id} out of range (0..{self.images.shape[0] - 1})"
                )
            result.append(self.images[img_id])
        return result
    
    def get_firing_rates(self, sample_index: int) -> np.ndarray:
        """Get firing rates for a specific sample.
        
        Args:
            sample_index: Index of the sample
            
        Returns:
            1D array of firing rates
            
        Raises:
            IndexError: If sample_index is out of range
            ValueError: If firing rates are not 1D
        """
        if sample_index < 0 or sample_index >= self.firing_rates.shape[0]:
            raise IndexError(
                f"sample_index {sample_index} out of range "
                f"(0..{self.firing_rates.shape[0] - 1})"
            )
        vec = self.firing_rates[sample_index]
        if vec.ndim != 1:
            raise ValueError("Loaded firing rates sample is not a 1D vector")
        return vec.astype(np.float32)


def _log_model_source_info(model_id: str) -> None:
    """Log run/experiment info for the model from MLflow registry.

    Logs the following (when available):
    - model_source_run_id
    - model_source_run_name
    - model_source_experiment_id
    - model_source_experiment_name
    """
    try:
        run_id = None

        try:
            versions = client.search_model_versions(f"name='{model_id}'")
        except Exception:
            versions = []

        if versions:
            preferred = None
            for mv in versions:
                if getattr(mv, "current_stage", "") == "Production":
                    preferred = mv
                    break
            if preferred is None:
                preferred = sorted(
                    versions,
                    key=lambda v: int(getattr(v, "version", 0)),
                    reverse=True,
                )[0]
            run_id = preferred.run_id

        if run_id is None:
            # Fallback: recent run that logged model artifacts
            try:
                df = mlflow.search_runs(
                    order_by=["attributes.start_time DESC"],
                    max_results=200,
                )
            except Exception:
                df = None
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    # Heuristic: keep the most recent run; stop at first
                    run_id = row.get("run_id") or row.get("info.run_id")
                    if run_id:
                        break

        if run_id is None:
            mlflow.log_param("model_source_info", "unavailable")
            return

        run = client.get_run(run_id)
        exp_id = run.info.experiment_id
        run_name = (
            run.data.tags.get("mlflow.runName")
            if run.data and run.data.tags
            else None
        )
        exp = client.get_experiment(exp_id)
        exp_name = exp.name if exp is not None else None

        mlflow.log_params(
            {
                "model_source_run_id": run_id,
                "model_source_run_name": run_name or "unknown",
                "model_source_experiment_id": exp_id,
                "model_source_experiment_name": exp_name or "unknown",
            }
        )
    except Exception as e:
        mlflow.log_param(
            "model_source_info_error", f"failed_to_log ({str(e)[:50]})"
        )


def _get_dataset_params_from_model(model_id: str) -> dict:
    """Get dataset parameters from a model ID or run ID.
    
    Args:
        model_id: Either a model registry ID (m-xxx) or a run ID
        
    Returns:
        Dictionary of parameters from the run
    """
    params = None
    run_id = None

    if len(model_id) == 32 and not model_id.startswith("m-"):
        print(f"[INFO] Input looks like a run ID, trying direct lookup: {model_id}")
        try:
            run = client.get_run(model_id)
            params = run.data.params
            run_id = model_id
            print(f"[INFO] Successfully retrieved run {run_id}")
        except Exception as e:
            print(f"[DEBUG] Failed to get run directly: {e}")

    if params is None:
        raise RuntimeError(
            f"Could not retrieve parameters for model/run {model_id}. "
            "Please provide a valid model ID or run ID."
        )
    
    # Log all dataset-related parameters for debugging
    dataset_params = {k: v for k, v in params.items() if 'dataset' in k.lower()}
    if dataset_params:
        print(f"[INFO] Dataset parameters found: {list(dataset_params.keys())}")
        #  print all the dataset params
        for k, v in dataset_params.items():
            print(f"    {k}: {v}")
    else:
        print("[WARNING] No dataset parameters found in run!")
    
    return params


def _detect_dataset_type(filename: str) -> str:
    """Detect dataset type from filename.
    
    Args:
        filename: Dataset filename
        
    Returns:
        Dataset type string (cifar10, cifar100, mnist, or unknown)
    """
    filename_lower = filename.lower()
    
    # Check for cifar10 first (more specific), then cifar, then mnist
    if "cifar10" in filename_lower or "cifar-10" in filename_lower:
        return "cifar10"
    elif "cifar100" in filename_lower or "cifar-100" in filename_lower:
        return "cifar100"
    elif "cifar" in filename_lower:
        return "cifar"
    elif "mnist" in filename_lower:
        return "mnist"
    else:
        return "unknown"


def _validate_dataset_type(detected_type: str, model_type: str, filename: str) -> None:
    """Validate that detected dataset type matches the model's training dataset type.
    
    Args:
        detected_type: Dataset type detected from filename
        model_type: Dataset type from model parameters
        filename: Dataset filename (for error messages)
        
    Raises:
        ValueError: If types don't match
    """
    if model_type == "unknown" or detected_type == "unknown":
        return
    
    # Normalize for comparison (cifar10 == cifar-10 == CIFAR10)
    model_type_normalized = model_type.lower().replace("-", "")
    detected_type_normalized = detected_type.lower().replace("-", "")
    
    if model_type_normalized != detected_type_normalized:
        raise ValueError(
            f"Dataset type mismatch! Model was trained on '{model_type}' "
            f"but trying to load '{detected_type}' dataset. "
            f"Dataset file: {filename}"
        )
    print(f"[INFO] ✓ Dataset type validated: {detected_type} matches model training dataset")


def _compute_original_image_loss(
    encoder: nn.Module,
    original_image: np.ndarray,
    target_rates: np.ndarray,
    loss_function: str,
    device: torch.device,
) -> float:
    """Compute the loss when passing the original image through the encoder.
    
    This represents the "best case" loss if reconstruction were perfect,
    and the delta between this and the final reconstruction loss shows
    how much room there is for improvement.
    
    Args:
        encoder: The encoder model in eval mode
        original_image: Original image as numpy array
        target_rates: Target firing rates as numpy array
        loss_function: Loss function type (mse, poisson_mean, poisson_sum)
        device: torch device to use
        
    Returns:
        Loss value as float
    """
    encoder.eval()
    
    # Convert original image to tensor and add batch dimension if needed
    img_tensor = torch.tensor(original_image.astype(np.float32)).to(device)
    if img_tensor.ndim == 2:  # (H, W)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif img_tensor.ndim == 3:  # (C, H, W) or (H, W, C)
        if img_tensor.shape[0] in [1, 3]:  # Assume (C, H, W)
            img_tensor = img_tensor.unsqueeze(0)  # (1, C, H, W)
        else:  # Assume (H, W, C)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    
    target_tensor = torch.tensor(target_rates.astype(np.float32)).to(device)
    
    # Setup loss function
    if loss_function == "mse":
        loss_fn = nn.MSELoss(reduction="mean")
    elif loss_function == "poisson_mean":
        loss_fn = nn.PoissonNLLLoss(log_input=False, reduction="mean")
    elif loss_function == "poisson_sum":
        loss_fn = nn.PoissonNLLLoss(log_input=False, reduction="sum")
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")
    
    softplus = nn.Softplus()
    
    with torch.no_grad():
        pred = encoder(img_tensor)
        if pred.ndim == 2 and pred.shape[0] == 1:
            pred = pred[0]
        elif pred.ndim != 1:
            raise ValueError("Encoder output must be shape (1, N) or (N,)")
        
        # Apply softplus for Poisson losses
        if loss_function in ["poisson_mean", "poisson_sum"]:
            pred = softplus(pred)
        
        loss = loss_fn(pred, target_tensor)
    
    return float(loss.cpu().item())



def _load_dataset_info(model_id: str) -> DatasetInfo:
    """Load and validate all dataset information for a model.
    
    This function consolidates all the repetitive dataset loading logic:
    - Gets model parameters from MLflow
    - Validates and loads the dataset file
    - Detects and validates dataset type
    - Returns a DatasetInfo object with all data cached
    
    Args:
        model_id: MLflow model ID or run ID
        
    Returns:
        DatasetInfo object containing all dataset information and data
        
    Raises:
        RuntimeError: If dataset filename not found in params
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If dataset type doesn't match model
    """
    print(f"[INFO] Loading dataset information for model: {model_id}")
    
    # 1. Get model parameters
    params = _get_dataset_params_from_model(model_id)
    
    # 2. Extract and validate dataset filename
    ds_filename = params.get("dataset_dataset_filename")
    if not ds_filename:
        raise RuntimeError(
            f"dataset_filename not found in run params for model {model_id}. "
            f"Available params: {list(params.keys())}. "
            "Cannot load dataset without knowing which dataset was used."
        )
    
    # 3. Construct and validate dataset path
    ds_path = os.path.join(
        get_path("workspace/datasets/synthetic"), "train", ds_filename
    )
    if not os.path.exists(ds_path):
        raise FileNotFoundError(
            f"Dataset file not found: {ds_path}. "
            "Sync datasets or provide a mirrored path."
        )
    
    # 4. Load dataset (both images and firing rates)
    print(f"[INFO] Loading dataset: {ds_filename}")
    images, firing_rates = load_npz_dataset(ds_path)
    print(f"[INFO] Dataset contains {images.shape[0]} images")
    print(f"[INFO] Image shape: {images.shape[1:]}")
    print(f"[INFO] Firing rates shape: {firing_rates.shape}")
    
    # 5. Detect and validate dataset type
    dataset_type = _detect_dataset_type(ds_filename)
    print(f"[INFO] Detected dataset type: {dataset_type}")
    
    model_dataset_type = params.get("dataset_dataset_type", "unknown")
    _validate_dataset_type(dataset_type, model_dataset_type, ds_filename)
    
    # 6. Create and return DatasetInfo object
    return DatasetInfo(
        model_id=model_id,
        params=params,
        filename=ds_filename,
        path=ds_path,
        dataset_type=dataset_type,
        images=images,
        firing_rates=firing_rates,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Input optimization with MLflow tracking",
    )
    p.add_argument(
        "--model-id",
        help="MLflow model ID to load the encoder",
    )
    p.add_argument(
        "--tracking-uri",
        help="MLflow tracking URI (e.g., file:./mlruns)",
    )
    p.add_argument(
        "--experiment-name",
        default="input_optimization",
        help="MLflow experiment name",
    )
    p.add_argument(
        "--run-name",
        help="Optional MLflow run name",
    )
    p.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index of sample to use when inferring target rates",
    )
    p.add_argument("--steps", type=int, default=2000, help="Steps to run")
    p.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    p.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Square image size (DEPRECATED: automatically inferred from dataset)",
    )
    p.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Num channels (only 1 supported; grayscale)",
    )
    p.add_argument("--log-every", type=int, default=50, help="Log cadence")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--output-dir",
        help="Where to save outputs; defaults to workspace/plots",
    )
    p.add_argument(
        "--image-ids",
        nargs="+",
        type=int,
        help="List of image IDs to reconstruct (e.g., --image-ids 0 1 2 10)",
    )
    p.add_argument(
        "--loss-function",
        choices=["mse", "poisson_mean", "poisson_sum"],
        default="poisson_mean",
        help="Loss function (default: poisson_mean)",
    )
    p.add_argument(
        "--scheduler",
        choices=["none", "cosine", "step", "exponential", "plateau"],
        default="cosine",
        help="Learning rate scheduler (default: cosine)",
    )
    p.add_argument(
        "--blur-sigma",
        type=float,
        default=2.5,
        help="Gaussian blur sigma for gradient smoothing (default: 2.5, use 0 to disable)",
    )
    return p


def main(args: argparse.Namespace) -> None:

    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S:%f") 
    
    # Set up MLflow tracking and artifact location
    from neurodecoders.config import get_base_path, get_mlflow_tracking_uri
    artifact_location = f"file://{get_base_path()}/mlruns"
    
    # Log configuration for debugging
    tracking_uri = args.tracking_uri if args.tracking_uri else get_mlflow_tracking_uri()
    print(f"[INFO] MLflow tracking URI: {tracking_uri}")
    print(f"[INFO] MLflow artifact location: {artifact_location}")
    print(f"[INFO] MLflow experiment: {args.experiment_name}")
    
    if args.tracking_uri:
        setup_mlflow_experiment(
            args.experiment_name, 
            tracking_uri=args.tracking_uri,
            artifact_location=artifact_location
        )
    else:
        setup_mlflow_experiment(
            args.experiment_name,
            artifact_location=artifact_location
        )

    run_name = args.run_name
    if not run_name:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"input_optim_{args.model_id}_{ts}"

    with mlflow.start_run(run_name=run_name, log_system_metrics=True):
        # Log info about the source model's MLflow run/experiment
        _log_model_source_info(args.model_id)

        # Load dataset info once (avoids loading dataset multiple times)
        print("[INFO] Loading dataset information...")
        dataset_info = _load_dataset_info(args.model_id)
        
        # Get image size from loaded dataset
        image_size = dataset_info.image_size
        
        # Warn if user provided image_size that doesn't match
        if args.image_size is not None and args.image_size != image_size:
            print(
                f"[WARNING] User provided --image-size {args.image_size}, "
                f"but dataset has images of size {image_size}. "
                f"Using dataset size {image_size} to avoid shape mismatch."
            )

        # Log params
        mlflow.log_params(
            {
                "model_id": args.model_id,
                "steps": args.steps,
                "lr": args.lr,
                "image_size": image_size,
                "image_size_source": "inferred_from_dataset",
                "channels": args.channels,
                "log_every": args.log_every,
                "seed": args.seed,
                "scheduler": args.scheduler,
                "blur_sigma": args.blur_sigma,
                "loss_function": args.loss_function,
            }
        )

        # Enforce grayscale only
        if args.channels != 1:
            raise ValueError(
                "Only grayscale is supported for input optimization. "
                "Please set --channels 1."
            )

        # Determine which images to reconstruct
        if args.image_ids:
            image_ids = args.image_ids
            mlflow.log_param("image_ids", image_ids)
            print(
                f"[DEBUG] Using multiple image reconstruction with IDs: "
                f"{image_ids}"
            )
        else:
            image_ids = [args.sample_index]
            mlflow.log_param("image_ids", image_ids)
            print(
                f"[DEBUG] Using single image reconstruction with ID: "
                f"{image_ids[0]}"
            )

        # Load encoder
        encoder = load_encoder_from_mlflow(args.model_id)

        # Configure optimizer with inferred image size
        cfg = OptimConfig(
            image_size=image_size,
            channels=args.channels,
            steps=args.steps,
            lr=args.lr,
            log_every=args.log_every,
            seed=args.seed,
            image_ids=image_ids,
            loss=args.loss_function,
            scheduler=args.scheduler,
            blur_sigma=args.blur_sigma,
        )

        # Set up output directory
        out_dir = (
            args.output_dir
            if args.output_dir is not None
            else get_path("workspace/plots/input_optim")
        )
        os.makedirs(out_dir, exist_ok=True)

        # Get target firing rates from dataset info
        target_raw: np.ndarray = dataset_info.get_firing_rates(image_ids[0])

        # Compute normalization stats from training set
        print("[INFO] Computing normalization statistics...")
        norm_stats = compute_normalization_stats(
            dataset_info.images, dataset_info.firing_rates
        )
        
        # Normalize the target firing rates
        firing_mean = np.array(norm_stats["firing_mean"])
        firing_std = np.array(norm_stats["firing_std"])
        # Avoid division by zero
        firing_std = np.where(firing_std == 0, 1.0, firing_std)
        
        target = (target_raw - firing_mean) / firing_std
        print(f"[INFO] Target normalized range: [{target.min():.4f}, {target.max():.4f}]")
        
        mlflow.log_param("target_source", "mlflow_dataset_normalized")
        mlflow.log_param("n_neurons", int(target.shape[0]))

        # Try to load original image for comparison
        try:
            print(f"[DEBUG] Loading original image for ID: {image_ids[0]}")
            original_images = dataset_info.get_images([image_ids[0]])
            print("[DEBUG] Successfully loaded original image")
            mlflow.log_param("original_image_loaded", True)
        except Exception as e:
            print(f"Warning: Could not load original image: {e}")
            original_images = None
            mlflow.log_param("original_image_loaded", False)

        optim_runner = ImageOptimizer(
            encoder=encoder, target_rates=target, config=cfg
        )
        img_np, metrics = optim_runner.optimize()

        # Compute additional metrics if original image is available
        if original_images:
            original_image = original_images[0]
            
            # 1. Compute SSIM between original and reconstructed image
            # Ensure images are in correct format for SSIM (2D grayscale)
            orig_2d = original_image.squeeze()
            recon_2d = img_np.squeeze()
            
            # SSIM expects data_range to be the range of the input images
            data_range = max(orig_2d.max() - orig_2d.min(), 
                           recon_2d.max() - recon_2d.min())
            
            ssim_value = ssim(orig_2d, recon_2d, data_range=data_range)
            metrics["ssim"] = float(ssim_value)
            print(f"[INFO] SSIM between original and reconstructed: {ssim_value:.4f}")
            
            # 2. Compute original image loss (baseline for comparison)
            device = torch.device("cuda" if torch.cuda.is_available() 
                                else "mps" if torch.backends.mps.is_available() 
                                else "cpu")
            
            original_loss = _compute_original_image_loss(
                encoder=encoder,
                original_image=original_image,
                target_rates=target,
                loss_function=args.loss_function,
                device=device,
            )
            metrics["original_image_loss"] = float(original_loss)
            
            # 3. Compute delta between reconstruction loss and original loss
            # Positive delta means reconstruction is worse (expected)
            # The larger the delta, the more room for improvement
            final_loss = metrics.get("loss", 0.0)
            loss_delta = final_loss - original_loss
            metrics["loss_delta"] = float(loss_delta)
            
            print(f"[INFO] Original image loss: {original_loss:.6f}")
            print(f"[INFO] Final reconstruction loss: {final_loss:.6f}")
            print(f"[INFO] Loss delta (room for improvement): {loss_delta:.6f}")

        # Create comparison plot using the new format
        from neurodecoders.input_optim.optimizer import (
            create_comparison_plots,
        )

        comparison_path = os.path.join(out_dir, f"comparison_plot_{image_ids[0]}_{date}.png")
        if original_images:
            create_comparison_plots(
                original_images=original_images,
                reconstructed_images=[img_np],
                image_ids=image_ids,
                output_path=comparison_path,
            )
            print(f"Single image comparison plot saved: {comparison_path}")
            log_single_artifact(comparison_path, artifact_path="images")
        

        # Log final metrics
        if metrics:
            mlflow.log_metrics(metrics)

        # Also save raw numpy for downstream use
        out_npy = os.path.join(out_dir, "reconstruction.npy")
        np.save(out_npy, img_np)
        log_single_artifact(out_npy, artifact_path="arrays")

        print("Optimization complete. Artifacts logged to MLflow.")
        if original_images:
            print(f"Comparison plot: {comparison_path}")
        else:
            # We saved the numpy array at out_npy above
            print(f"Saved reconstruction numpy: {out_npy}")



if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
