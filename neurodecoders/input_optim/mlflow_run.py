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
    p.add_argument("--channels", type=int, default=1, help="Num channels")
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

        # Load target rates inferred from MLflow model's dataset
        target: np.ndarray = _infer_target_rates_from_model(
            model_id=args.model_id, sample_index=args.sample_index
        )
        mlflow.log_param("target_source", "mlflow_dataset")
        mlflow.log_param("n_neurons", int(target.shape[0]))

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
        )

        optim_runner = ImageOptimizer(
            encoder=encoder, target_rates=target, config=cfg
        )
        img_np, metrics = optim_runner.optimize()

        # Save image and log artifact
        out_dir = (
            args.output_dir
            if args.output_dir is not None
            else get_path("workspace/plots/input_optim")
        )
        os.makedirs(out_dir, exist_ok=True)
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


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
