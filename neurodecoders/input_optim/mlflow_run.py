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

DEFAULT_MODEL_ID = "m-553e4f38555b44a6a026362915f9431c"


def _log_model_source_info(model_id: str) -> None:
    """Log run/experiment info for the model from MLflow registry.

    Logs the following (when available):
    - model_source_run_id
    - model_source_run_name
    - model_source_experiment_id
    - model_source_experiment_name
    """
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
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


def _load_original_images_from_dataset(
    model_id: str, image_ids: List[int]
) -> List[np.ndarray]:
    """Load original images from the model's training dataset.

    Args:
        model_id: MLflow model ID
        image_ids: List of image indices to load

    Returns:
        List of original images as numpy arrays
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    # Get dataset path from model's training run
    params = None
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
        run = client.get_run(run_id)
        params = run.data.params
    else:
        # Fallback: latest run with dataset params
        try:
            df = mlflow.search_runs(
                order_by=["attributes.start_time DESC"], max_results=200
            )
        except Exception:
            df = None
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                ds = row.get("params.dataset_dataset_filename")
                if isinstance(ds, str) and ds:
                    params = {"dataset_dataset_filename": ds}
                    break

    if params is None:
        # Fallback: pick the most recent synthetic train dataset file
        train_dir = os.path.join(
            get_path("workspace/datasets/synthetic"), "train"
        )
        if not os.path.exists(train_dir):
            raise FileNotFoundError(
                f"No train dir found at {train_dir} and no MLflow "
                "runs with dataset params."
            )
        cand = [f for f in os.listdir(train_dir) if f.endswith(".npz")]
        if not cand:
            raise FileNotFoundError(
                "No .npz datasets in synthetic/train and no MLflow "
                "runs with dataset params."
            )
        latest = max(
            cand,
            key=lambda fn: os.path.getmtime(os.path.join(train_dir, fn)),
        )
        params = {"dataset_dataset_filename": latest}

    ds_filename = params.get("dataset_dataset_filename")
    if not ds_filename:
        raise RuntimeError(
            "dataset_filename not found in run params. "
            "Cannot load original images."
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
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    params = None
    # First try model registry
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
        run = client.get_run(run_id)
        params = run.data.params
    else:
        # Fallback: latest run with dataset params using fluent API
        try:
            df = mlflow.search_runs(
                order_by=["attributes.start_time DESC"], max_results=200
            )
        except Exception:
            df = None
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                ds = row.get("params.dataset_dataset_filename")
                if isinstance(ds, str) and ds:
                    params = {
                        "dataset_dataset_filename": ds,
                    }
                    break
        if params is None:
            # Fallback: pick the most recent synthetic train dataset file
            train_dir = os.path.join(
                get_path("workspace/datasets/synthetic"), "train"
            )
            if not os.path.exists(train_dir):
                raise FileNotFoundError(
                    f"No train dir found at {train_dir} and no MLflow "
                    "runs with dataset params."
                )
            cand = [f for f in os.listdir(train_dir) if f.endswith(".npz")]
            if not cand:
                raise FileNotFoundError(
                    "No .npz datasets in synthetic/train and no MLflow "
                    "runs with dataset params."
                )
            latest = max(
                cand,
                key=lambda fn: os.path.getmtime(os.path.join(train_dir, fn)),
            )
            params = {"dataset_dataset_filename": latest}

    # Dataset parameters were logged with 'dataset_*' keys
    ds_filename = params.get("dataset_dataset_filename")
    if not ds_filename:
        raise RuntimeError(
            "dataset_filename not found in run params. Cannot infer dataset."
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
        default=DEFAULT_MODEL_ID,
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
    p.add_argument(
        "--tv-weight",
        type=float,
        default=1e-4,
        help="Total variation",
    )
    p.add_argument("--l2-weight", type=float, default=1e-6, help="L2 weight")
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
    return p


def main(args: argparse.Namespace) -> None:
    if args.tracking_uri:
        setup_mlflow_experiment(args.experiment_name, args.tracking_uri)
    else:
        setup_mlflow_experiment(args.experiment_name)

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
                "tv_weight": args.tv_weight,
                "l2_weight": args.l2_weight,
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
            tv_weight=args.tv_weight,
            l2_weight=args.l2_weight,
            log_every=args.log_every,
            seed=args.seed,
            image_ids=image_ids,
        )

        # Set up output directory
        out_dir = (
            args.output_dir
            if args.output_dir is not None
            else get_path("workspace/plots/input_optim")
        )
        os.makedirs(out_dir, exist_ok=True)

        if len(image_ids) == 1:
            # Single image reconstruction (original behavior)
            target: np.ndarray = _infer_target_rates_from_model(
                model_id=args.model_id, sample_index=image_ids[0]
            )
            mlflow.log_param("target_source", "mlflow_dataset")
            mlflow.log_param("n_neurons", int(target.shape[0]))

            optim_runner = ImageOptimizer(
                encoder=encoder, target_rates=target, config=cfg
            )
            img_np, metrics = optim_runner.optimize()

            # Save image and log artifact
            out_img = os.path.join(out_dir, "reconstruction.png")
            optim_runner.save_image(out_img)
            log_single_artifact(out_img, artifact_path="images")
            try:
                art_uri = mlflow.get_artifact_uri("images/reconstruction.png")
            except Exception:
                art_uri = None

            # Log final metrics
            if metrics:
                mlflow.log_metrics(metrics)

            # Also save raw numpy for downstream use
            out_npy = os.path.join(out_dir, "reconstruction.npy")
            np.save(out_npy, img_np)
            log_single_artifact(out_npy, artifact_path="arrays")

            print("Optimization complete. Artifacts logged to MLflow.")
            print(f"Saved image: {out_img}")
            if art_uri:
                print(f"MLflow artifact: {art_uri}")
        else:
            # Multiple image reconstruction
            target_rates_list = []
            for img_id in image_ids:
                target_rates = _infer_target_rates_from_model(
                    model_id=args.model_id, sample_index=img_id
                )
                target_rates_list.append(target_rates)

            # Load original images for comparison
            try:
                print(f"[DEBUG] Loading original images for IDs: {image_ids}")
                original_images = _load_original_images_from_dataset(
                    model_id=args.model_id, image_ids=image_ids
                )
                print(
                    f"[DEBUG] Successfully loaded "
                    f"{len(original_images)} original images"
                )
                mlflow.log_param("original_images_loaded", True)
            except Exception as e:
                print(f"Warning: Could not load original images: {e}")
                original_images = None
                mlflow.log_param("original_images_loaded", False)

            mlflow.log_param("target_source", "mlflow_dataset")
            mlflow.log_param("n_neurons", int(target_rates_list[0].shape[0]))
            mlflow.log_param("n_images", len(image_ids))

            # Create optimizer instance for multiple reconstructions
            optim_runner = ImageOptimizer(
                encoder=encoder, target_rates=target_rates_list[0], config=cfg
            )

            # Reconstruct multiple images
            original_images, reconstructed_images = (
                optim_runner.reconstruct_multiple_images(
                    target_rates_list=target_rates_list,
                    image_ids=image_ids,
                    output_dir=out_dir,
                    original_images=original_images,
                )
            )

            # Log all artifacts
            for i, img_id in enumerate(image_ids):
                img_path = os.path.join(
                    out_dir, f"reconstruction_{img_id}.png"
                )
                npy_path = os.path.join(
                    out_dir, f"reconstruction_{img_id}.npy"
                )

                if os.path.exists(img_path):
                    log_single_artifact(
                        img_path,
                        artifact_path=f"images/reconstruction_{img_id}.png",
                    )
                if os.path.exists(npy_path):
                    log_single_artifact(
                        npy_path,
                        artifact_path=f"arrays/reconstruction_{img_id}.npy",
                    )

            # Log comparison plot if it exists
            comparison_path = os.path.join(out_dir, "comparison_plot.png")
            if os.path.exists(comparison_path):
                log_single_artifact(
                    comparison_path, artifact_path="images/comparison_plot.png"
                )

            print(
                f"Multiple image reconstruction complete. {len(image_ids)} "
                "images processed."
            )
            print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
