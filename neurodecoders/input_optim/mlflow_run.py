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
from datetime import datetime
from typing import List

import mlflow
import numpy as np

from neurodecoders.data.loading import load_npz_dataset
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

    # 1) If looks like a run ID (32 hex chars, no m- prefix), try direct lookup
    if len(model_id) == 32 and not model_id.startswith("m-"):
        print(f"[INFO] Input looks like a run ID, trying direct lookup: {model_id}")
        try:
            run = client.get_run(model_id)
            params = run.data.params
            run_id = model_id
            print(f"[INFO] Successfully retrieved run {run_id}")
        except Exception as e:
            print(f"[DEBUG] Failed to get run directly: {e}")

    # 2) If still not found and model_id looks like a model registry name (m-... or plain name),
    # try searching model registry versions for a linked run_id
    if params is None:
        try:
            print(f"[INFO] Trying model registry lookup for: {model_id}")
            versions = client.search_model_versions(f"name='{model_id}'")
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
                run_id = getattr(preferred, "run_id", None) or getattr(preferred, "run_id", None)
                if run_id:
                    try:
                        run = client.get_run(run_id)
                        params = run.data.params
                        print(f"[INFO] Retrieved params from model registry version run {run_id}")
                    except Exception as e:
                        print(f"[DEBUG] Could not get run {run_id} from registry entry: {e}")
        except Exception as e:
            print(f"[DEBUG] Model registry lookup failed: {e}")

    # 3) Final fallback: search recent runs and try to match by model_id in params or tags
    if params is None:
        try:
            print("[INFO] Searching recent MLflow runs for candidate runs to extract dataset params")
            df = mlflow.search_runs(order_by=["attributes.start_time DESC"], max_results=500)
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    # If run_id matches exactly, use it
                    candidate_run_id = row.get("run_id") or row.get("info.run_id")
                    if candidate_run_id and candidate_run_id == model_id:
                        try:
                            run = client.get_run(candidate_run_id)
                            params = run.data.params
                            run_id = candidate_run_id
                            print(f"[INFO] Matched run_id exactly: {run_id}")
                            break
                        except Exception:
                            continue
                    # Otherwise check if the run logged a 'model_id' param equal to provided model_id
                    try:
                        rid = row.get("params.model_id") or row.get("data.params.model_id")
                    except Exception:
                        rid = None
                    if rid and str(rid) == str(model_id):
                        candidate_run_id = row.get("run_id") or row.get("info.run_id")
                        try:
                            run = client.get_run(candidate_run_id)
                            params = run.data.params
                            run_id = candidate_run_id
                            print(f"[INFO] Found run with matching 'model_id' param: {run_id}")
                            break
                        except Exception:
                            continue
        except Exception as e:
            print(f"[DEBUG] Recent runs search failed: {e}")

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


def _load_original_images_from_dataset(
    model_id: str, image_ids: List[int]
) -> List[np.ndarray]:
    """Load original images from the model's training dataset.

    Args:
        model_id: MLflow model ID or run ID
        image_ids: List of image indices to load

    Returns:
        List of original images as numpy arrays
    """
    # Get params using the helper function
    params = _get_dataset_params_from_model(model_id)

    ds_filename = params.get("dataset_dataset_filename")
    if not ds_filename:
        raise RuntimeError(
            f"dataset_filename not found in run params for model {model_id}. "
            f"Available params: {list(params.keys())}. "
            "Cannot load original images without knowing which dataset was used."
        )

    ds_path = os.path.join(
        get_path("workspace/datasets/synthetic"), "train", ds_filename
    )
    if not os.path.exists(ds_path):
        raise FileNotFoundError(
            f"Dataset file not found: {ds_path}. "
            "Sync datasets or provide a mirrored path."
        )

    # Load dataset
    images, _ = load_npz_dataset(ds_path)
    
    # Log dataset information for verification
    print(f"[INFO] Loaded dataset: {ds_filename}")
    print(f"[INFO] Dataset contains {images.shape[0]} images")
    print(f"[INFO] Image shape: {images.shape[1:]}")
    
    # Extract dataset type from filename for validation
    # Check for cifar10 first (more specific), then cifar, then mnist
    if "cifar10" in ds_filename.lower() or "cifar-10" in ds_filename.lower():
        dataset_type = "cifar10"
    elif "cifar100" in ds_filename.lower() or "cifar-100" in ds_filename.lower():
        dataset_type = "cifar100"
    elif "cifar" in ds_filename.lower():
        dataset_type = "cifar"
    elif "mnist" in ds_filename.lower():
        dataset_type = "mnist"
    else:
        dataset_type = "unknown"
    print(f"[INFO] Detected dataset type: {dataset_type}")
    
    # Validate dataset type matches model if available in params
    model_dataset_type = params.get("dataset_dataset_type", "unknown")
    if model_dataset_type != "unknown" and dataset_type != "unknown":
        # Normalize for comparison (cifar10 == cifar-10 == CIFAR10)
        model_type_normalized = model_dataset_type.lower().replace("-", "")
        dataset_type_normalized = dataset_type.lower().replace("-", "")
        
        if model_type_normalized != dataset_type_normalized:
            raise ValueError(
                f"Dataset type mismatch! Model was trained on '{model_dataset_type}' "
                f"but trying to load '{dataset_type}' dataset. "
                f"Dataset file: {ds_filename}"
            )
        print(f"[INFO] ✓ Dataset type validated: {dataset_type} matches model training dataset")

    # Extract requested images
    original_images = []
    for img_id in image_ids:
        if img_id < 0 or img_id >= images.shape[0]:
            raise IndexError(
                f"image_id {img_id} out of range (0..{images.shape[0] - 1})"
            )
        original_images.append(images[img_id])

    return original_images


def _infer_target_rates_from_model(
    model_id: str, sample_index: int
) -> np.ndarray:
    """Infer a target firing-rate vector from the model's training dataset.

    Uses MLflow's model registry to locate the model's originating run,
    reads dataset parameters logged during training (including
    dataset_filename), loads the synthetic train .npz, and returns the
    firing-rate vector at the requested sample index.
    """
    # Get params using the helper function
    params = _get_dataset_params_from_model(model_id)

    # Dataset parameters were logged with 'dataset_*' keys
    ds_filename = params.get("dataset_dataset_filename")
    if not ds_filename:
        raise RuntimeError(
            f"dataset_filename not found in run params for model {model_id}. "
            f"Available params: {list(params.keys())}. "
            "Cannot infer dataset without knowing which dataset was used."
        )

    ds_path = os.path.join(
        get_path("workspace/datasets/synthetic"), "train", ds_filename
    )
    if not os.path.exists(ds_path):
        raise FileNotFoundError(
            f"Dataset file not found on this machine: {ds_path}. "
            "Sync datasets or provide a mirrored path."
        )

    _, firing = load_npz_dataset(ds_path)
    
    # Log dataset information for verification
    print(f"[INFO] Loaded dataset for target rates: {ds_filename}")
    print(f"[INFO] Dataset contains {firing.shape[0]} samples with {firing.shape[1]} neurons")
    
    # Extract dataset type from filename for validation
    # Check for cifar10 first (more specific), then cifar, then mnist
    if "cifar10" in ds_filename.lower() or "cifar-10" in ds_filename.lower():
        dataset_type = "cifar10"
    elif "cifar100" in ds_filename.lower() or "cifar-100" in ds_filename.lower():
        dataset_type = "cifar100"
    elif "cifar" in ds_filename.lower():
        dataset_type = "cifar"
    elif "mnist" in ds_filename.lower():
        dataset_type = "mnist"
    else:
        dataset_type = "unknown"
    print(f"[INFO] Detected dataset type: {dataset_type}")
    
    # Validate dataset type matches model if available in params
    model_dataset_type = params.get("dataset_dataset_type", "unknown")
    if model_dataset_type != "unknown" and dataset_type != "unknown":
        # Normalize for comparison (cifar10 == cifar-10 == CIFAR10)
        model_type_normalized = model_dataset_type.lower().replace("-", "")
        dataset_type_normalized = dataset_type.lower().replace("-", "")
        
        if model_type_normalized != dataset_type_normalized:
            raise ValueError(
                f"Dataset type mismatch! Model was trained on '{model_dataset_type}' "
                f"but trying to load '{dataset_type}' dataset. "
                f"Dataset file: {ds_filename}"
            )
        print(f"[INFO] ✓ Dataset type validated: {dataset_type} matches model training dataset")
    
    if sample_index < 0 or sample_index >= firing.shape[0]:

        raise IndexError(
            f"sample_index {sample_index} out of range "
            f"(0..{firing.shape[0] - 1})"
        )
    vec = firing[sample_index]
    if vec.ndim != 1:
        raise ValueError("Loaded firing rates sample is not a 1D vector")
    return vec.astype(np.float32)


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
        default=64,
        help="Square image size",
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

        # Log params
        mlflow.log_params(
            {
                "model_id": args.model_id,
                "steps": args.steps,
                "lr": args.lr,
                "image_size": args.image_size,
                "channels": args.channels,
                "log_every": args.log_every,
                "seed": args.seed,
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

        # Configure optimizer
        cfg = OptimConfig(
            image_size=args.image_size,
            channels=args.channels,
            steps=args.steps,
            lr=args.lr,
            log_every=args.log_every,
            seed=args.seed,
            image_ids=image_ids,
            loss=args.loss_function,
        )

        # Set up output directory
        out_dir = (
            args.output_dir
            if args.output_dir is not None
            else get_path("workspace/plots/input_optim")
        )
        os.makedirs(out_dir, exist_ok=True)

        # Single image reconstruction with comparison plot
        target: np.ndarray = _infer_target_rates_from_model(
            model_id=args.model_id, sample_index=image_ids[0]
        )
        mlflow.log_param("target_source", "mlflow_dataset")
        mlflow.log_param("n_neurons", int(target.shape[0]))

        # Try to load original image for comparison
        try:
            print(f"[DEBUG] Loading original image for ID: {image_ids[0]}")
            original_images = _load_original_images_from_dataset(
                model_id=args.model_id, image_ids=[image_ids[0]]
            )
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
