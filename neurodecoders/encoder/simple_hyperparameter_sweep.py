#!/usr/bin/env python3
"""
Simplified hyperparameter sweep for neural encoders.

This script reuses existing training infrastructure to run hyperparameter
sweeps without code duplication.
"""

import argparse
import itertools
from typing import Any, Dict, List, Optional, cast

import mlflow
import numpy as np

from neurodecoders.encoder.config import HYPERPARAMETER_SWEEP_DEFAULTS
from neurodecoders.encoder.mlflow_training import train_with_config


def generate_combinations(
    learning_rates: Optional[List[float]] = None,
    batch_sizes: Optional[List[int]] = None,
    optimizers: Optional[List[str]] = None,
    weight_decays: Optional[List[float]] = None,
    schedulers: Optional[List[str]] = None,
    loss_functions: Optional[List[str]] = None,
    model_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Generate hyperparameter combinations."""
    # Use defaults if not provided
    learning_rates = learning_rates or cast(
        List[float], HYPERPARAMETER_SWEEP_DEFAULTS["learning_rates"]
    )
    batch_sizes = batch_sizes or cast(
        List[int], HYPERPARAMETER_SWEEP_DEFAULTS["batch_sizes"]
    )
    optimizers = optimizers or cast(
        List[str], HYPERPARAMETER_SWEEP_DEFAULTS["optimizers"]
    )
    weight_decays = weight_decays or cast(
        List[float], HYPERPARAMETER_SWEEP_DEFAULTS["weight_decay"]
    )
    schedulers = schedulers or cast(
        List[str], HYPERPARAMETER_SWEEP_DEFAULTS["schedulers"]
    )
    loss_functions = loss_functions or cast(
        List[str], HYPERPARAMETER_SWEEP_DEFAULTS["loss_functions"]
    )
    model_types = model_types or cast(
        List[str], HYPERPARAMETER_SWEEP_DEFAULTS["sweep_model_type"]
    )

    # Generate combinations
    combinations = list(
        itertools.product(
            learning_rates,
            batch_sizes,
            optimizers,
            weight_decays,
            schedulers,
            loss_functions,
            model_types,
        )
    )

    # Convert to configs
    configs = []
    for lr, bs, opt, wd, sched, loss_fn, model_type in combinations:
        config = {
            "learning_rate": lr,
            "batch_size": bs,
            "optimizer": opt,
            "weight_decay": wd,
            "scheduler": sched,
            "loss_function": loss_fn,
            "model_type": model_type,
        }
        configs.append(config)

    return configs


def run_sweep(
    configs: List[Dict[str, Any]],
    base_config: Dict[str, Any],
    experiment_name: str = "hyperparameter_sweep",
    max_runs: Optional[int] = None,
    random_seed: int = 42,
):
    """Run hyperparameter sweep using existing training infrastructure."""
    np.random.seed(random_seed)

    # Limit runs if specified
    if max_runs and len(configs) > max_runs:
        print(
            f"Limiting to {max_runs} runs from {len(configs)} configurations"
        )
        indices = np.random.choice(len(configs), max_runs, replace=False)
        configs = [configs[i] for i in indices]

    print(f"Running sweep with {len(configs)} configurations")

    # Set up MLflow experiment
    mlflow.set_experiment(experiment_name)

    results = []

    for i, config in enumerate(configs):
        print(f"\n=== Run {i + 1}/{len(configs)} ===")
        print(f"Config: {config}")

        # Merge with base config
        full_config = {**base_config, **config}

        # Create run name
        run_name = (
            f"{config['model_type']}_lr{config['learning_rate']}_"
            f"bs{config['batch_size']}_{config['optimizer']}_"
            f"wd{config['weight_decay']}_{config['scheduler']}_"
            f"{config['loss_function']}"
        )

        full_config["mlflow_run_name"] = run_name

        # Start MLflow run
        with mlflow.start_run(run_name=run_name):
            try:
                # Use existing training function
                trainer, lightning_model, _ = train_with_config(full_config)

                # Log results
                result = {
                    "run_name": run_name,
                    "config": config,
                    "final_train_loss": lightning_model.train_losses[-1]
                    if lightning_model.train_losses
                    else None,
                    "final_val_loss": lightning_model.val_losses[-1]
                    if lightning_model.val_losses
                    else None,
                    "best_epoch": len(lightning_model.val_losses)
                    if lightning_model.val_losses
                    else None,
                }

                # Log to MLflow
                mlflow.log_params(config)
                mlflow.log_metrics(
                    {
                        "final_train_loss": result["final_train_loss"],
                        "final_val_loss": result["final_val_loss"],
                        "best_epoch": result["best_epoch"],
                    }
                )

                results.append(result)

                print(f"Completed: {run_name}")
                print(f"Final val loss: {result['final_val_loss']:.4f}")

            except Exception as e:
                print(f"Error in run {run_name}: {e}")
                mlflow.log_param("error", str(e))
                continue

    # Print summary
    print("\n=== Sweep Summary ===")
    print(f"Total runs: {len(results)}")

    if results:
        best_result = min(
            results,
            key=lambda x: cast(
                float,
                x["final_val_loss"]
                if x["final_val_loss"] is not None
                else float("inf"),
            ),
        )
        print(f"Best configuration: {best_result['run_name']}")
        print(f"Best validation loss: {best_result['final_val_loss']:.4f}")

        # Top 5 configurations
        sorted_results = sorted(
            results,
            key=lambda x: cast(
                float,
                x["final_val_loss"]
                if x["final_val_loss"] is not None
                else float("inf"),
            ),
        )
        print("\nTop 5 configurations:")
        for i, result in enumerate(sorted_results[:5]):
            print(
                f"{i + 1}. {result['run_name']}: "
                f"{result['final_val_loss']:.4f}"
            )

    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep")

    # Base configuration
    parser.add_argument(
        "--dataset-type", default="cifar10", help="Dataset type"
    )
    parser.add_argument(
        "--sta-type",
        default="periodic_patterns,70,70",
        help="STA pattern type",
    )
    parser.add_argument(
        "--n-neurons", type=int, default=100, help="Number of neurons"
    )
    parser.add_argument(
        "--n-images", type=int, default=10000, help="Number of images"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs"
    )

    # Sweep options
    parser.add_argument(
        "--learning-rates", nargs="+", type=float, help="Learning rates to try"
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, help="Batch sizes to try"
    )
    parser.add_argument("--optimizers", nargs="+", help="Optimizers to try")
    parser.add_argument(
        "--weight-decays", nargs="+", type=float, help="Weight decay values"
    )
    parser.add_argument("--schedulers", nargs="+", help="Schedulers to try")
    parser.add_argument(
        "--loss-functions", nargs="+", help="Loss functions to try"
    )
    parser.add_argument("--model-types", nargs="+", help="Model types to try")

    # Control
    parser.add_argument(
        "--experiment-name",
        default="hyperparameter_sweep",
        help="MLflow experiment name",
    )
    parser.add_argument("--max-runs", type=int, help="Maximum number of runs")
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed"
    )

    args = parser.parse_args()

    # Base configuration
    base_config = {
        "dataset_type": args.dataset_type,
        "sta_type": args.sta_type,
        "n_neurons": args.n_neurons,
        "n_images": args.n_images,
        "epochs": args.epochs,
        "enable_mixed_precision": True,
        "enable_early_stopping": True,
        "early_stopping_patience": 100,  # Updated default
        "enable_checkpointing": True,
        "gradient_clip_val": 1.0,
    }

    # Generate combinations
    configs = generate_combinations(
        learning_rates=args.learning_rates,
        batch_sizes=args.batch_sizes,
        optimizers=args.optimizers,
        weight_decays=args.weight_decays,
        schedulers=args.schedulers,
        loss_functions=args.loss_functions,
        model_types=args.model_types,
    )

    print(f"Generated {len(configs)} hyperparameter combinations")

    # Run sweep
    results = run_sweep(
        configs=configs,
        base_config=base_config,
        experiment_name=args.experiment_name,
        max_runs=args.max_runs,
        random_seed=args.random_seed,
    )

    print(
        f"Hyperparameter sweep completed with {len(results)} successful runs"
    )


if __name__ == "__main__":
    main()
