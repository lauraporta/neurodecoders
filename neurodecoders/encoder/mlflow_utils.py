"""
MLflow utilities for neural encoder experiment tracking.

This module provides MLflow integration for tracking experiments,
hyperparameters, metrics, and model artifacts during encoder training.
"""

import os
from typing import Any, Dict, Optional

import mlflow
import mlflow.pytorch
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import MLFlowLogger


class MLflowExperimentTracker:
    """
    MLflow experiment tracker for neural encoder training.

    This class provides a unified interface for tracking experiments,
    logging hyperparameters, metrics, and model artifacts.
    """

    def __init__(
        self,
        experiment_name: str = "neural_encoder",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ):
        """
        Initialize MLflow experiment tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (optional)
            artifact_location: Location for storing artifacts (optional)
        """
        self.experiment_name = experiment_name

        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Set up experiment
        mlflow.set_experiment(experiment_name)

        # Set artifact location if provided
        if artifact_location:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(
                    experiment_name, artifact_location=artifact_location
                )

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Start a new MLflow run.

        Args:
            run_name: Name for this specific run
            tags: Dictionary of tags to add to the run
        """
        return mlflow.start_run(run_name=run_name, tags=tags)

    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """
        Log hyperparameters for the current run.

        Args:
            hyperparams: Dictionary of hyperparameters to log
        """
        mlflow.log_params(hyperparams)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ):
        """
        Log metrics for the current run.

        Args:
            metrics: Dictionary of metrics to log
            step: Step number for the metrics
        """
        mlflow.log_metrics(metrics, step=step)

    def log_model(
        self,
        model: LightningModule,
        model_name: str = "encoder_model",
        registered_model_name: Optional[str] = None,
    ):
        """
        Log a PyTorch Lightning model.

        Args:
            model: The Lightning module to log
            model_name: Name for the model artifact
            registered_model_name: Name for model registry (optional)
        """
        mlflow.pytorch.log_model(
            model,
            artifact_path=model_name,
            registered_model_name=registered_model_name,
        )

    def log_artifacts(
        self, local_dir: str, artifact_path: Optional[str] = None
    ):
        """
        Log artifacts from a local directory.

        Args:
            local_dir: Local directory containing artifacts
            artifact_path: Path within the run's artifact directory
        """
        mlflow.log_artifacts(local_dir, artifact_path)

    def log_model_metadata(
        self,
        model_path: str,
        model_type: str,
        dataset_info: Dict[str, Any],
        training_info: Dict[str, Any],
    ):
        """
        Log model metadata as a JSON artifact.

        Args:
            model_path: Path to the saved model
            model_type: Type of model (e.g., 'simple_encoder',
            'resnet_encoder')
            dataset_info: Information about the dataset used
            training_info: Information about the training process
        """
        import json

        metadata = {
            "model_type": model_type,
            "model_path": model_path,
            "dataset_info": dataset_info,
            "training_info": training_info,
            "timestamp": mlflow.active_run().info.start_time,
        }

        # Save metadata to temporary file
        metadata_path = "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Log as artifact
        mlflow.log_artifact(metadata_path)

        # Clean up
        os.remove(metadata_path)

    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()


def create_mlflow_logger(
    experiment_name: str = "neural_encoder",
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    log_model: bool = True,
) -> MLFlowLogger:
    """
    Create an MLflow logger for PyTorch Lightning.

    Args:
        experiment_name: Name of the MLflow experiment
        run_name: Name for this specific run
        tracking_uri: MLflow tracking server URI
        log_model: Whether to log the model automatically

    Returns:
        MLFlowLogger instance
    """
    return MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        log_model=log_model,
    )


def log_encoder_experiment(
    model: torch.nn.Module,
    lightning_module: LightningModule,
    hyperparams: Dict[str, Any],
    dataset_info: Dict[str, Any],
    model_save_path: str,
    experiment_name: str = "neural_encoder",
    run_name: Optional[str] = None,
):
    """
    Comprehensive logging function for encoder experiments.

    Args:
        model: The trained model
        lightning_module: The Lightning module
        hyperparams: Training hyperparameters
        dataset_info: Information about the dataset
        model_save_path: Path where model was saved
        experiment_name: MLflow experiment name
        run_name: Name for this run
    """
    tracker = MLflowExperimentTracker(experiment_name)

    with tracker.start_run(run_name=run_name):
        # Log hyperparameters
        tracker.log_hyperparameters(hyperparams)

        # Log dataset info
        tracker.log_hyperparameters(
            {f"dataset_{k}": v for k, v in dataset_info.items()}
        )

        # Log model
        tracker.log_model(lightning_module, "encoder_model")

        # Log model metadata
        training_info = {
            "total_epochs": len(lightning_module.train_losses),
            "final_train_loss": lightning_module.train_losses[-1]
            if lightning_module.train_losses
            else None,
            "final_val_loss": lightning_module.val_losses[-1]
            if lightning_module.val_losses
            else None,
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        }

        tracker.log_model_metadata(
            model_save_path,
            type(model).__name__,
            dataset_info,
            training_info,
        )

        # Log training curves as artifacts
        if lightning_module.train_losses and lightning_module.val_losses:
            import datetime

            import matplotlib.pyplot as plt

            # Create workspace plots directory if it doesn't exist
            plots_dir = "workspace/plots/training"
            os.makedirs(plots_dir, exist_ok=True)

            plt.figure(figsize=(10, 6))
            plt.plot(lightning_module.train_losses, label="Train Loss")
            plt.plot(lightning_module.val_losses, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Curves")
            plt.legend()
            plt.grid(True)

            # Save plot to workspace with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_type = type(model).__name__
            plot_filename = f"training_curves_{model_type}_{timestamp}.png"
            plot_path = os.path.join(plots_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Training curves saved to: {plot_path}")

            # Also log to MLflow
            tracker.log_artifacts(plots_dir, "plots")


def get_experiment_comparison(experiment_name: str = "neural_encoder"):
    """
    Get a comparison of all runs in an experiment.

    Args:
        experiment_name: Name of the experiment

    Returns:
        DataFrame with run comparisons
    """
    import pandas as pd

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame()

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], output_format="pandas"
    )

    return runs
