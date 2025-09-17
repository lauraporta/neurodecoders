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
        "--target-rates-npy",
        required=True,
        help="Path to a NumPy .npy file with target rates, shape (N,)",
    )
    p.add_argument("--steps", type=int, default=2000, help="Steps to run")
    p.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    p.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Square image size",
    )
    p.add_argument("--channels", type=int, default=3, help="Num channels")
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

        # Load target rates
        if not os.path.exists(args.target_rates_npy):
            raise FileNotFoundError(args.target_rates_npy)
        target = np.load(args.target_rates_npy)
        if target.ndim != 1:
            raise ValueError("target_rates must be 1D (N,)")
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

        # Log final metrics
        if metrics:
            mlflow.log_metrics(metrics)

        # Also save raw numpy for downstream use
        out_npy = os.path.join(out_dir, "reconstruction.npy")
        np.save(out_npy, img_np)
        log_single_artifact(out_npy, artifact_path="arrays")

        print("Optimization complete. Artifacts logged to MLflow.")


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
