"""
MLflow utilities for neural encoder experiment tracking.

This module provides MLflow integration for tracking experiments,
hyperparameters, metrics, and model artifacts during encoder training.
"""

import json
import os
from typing import Any, Dict, Optional

import mlflow
import mlflow.pytorch
import pandas as pd
from pytorch_lightning import LightningModule


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

        # Set up experiment - handle deleted experiments gracefully
        try:
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(
                f"Warning: Could not set experiment '{experiment_name}': {e}"
            )
            print("Creating new experiment...")
            try:
                mlflow.create_experiment(
                    experiment_name, artifact_location=artifact_location
                )
                mlflow.set_experiment(experiment_name)
            except Exception as e2:
                print(f"Error creating experiment: {e2}")
                # Fall back to default experiment
                mlflow.set_experiment("Default")

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
        # Check if there's an active run and end it if necessary
        try:
            active_run = mlflow.active_run()
            if active_run is not None:
                print(f"Ending active run: {active_run.info.run_id}")
                mlflow.end_run()
        except Exception as e:
            print(f"Warning: Could not check/end active run: {e}")

        return mlflow.start_run(
            run_name=run_name, tags=tags, log_system_metrics=True
        )

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


def get_experiment_comparison(experiment_name: str = "neural_encoder"):
    """
    Get a comparison of all runs in an experiment.

    Args:
        experiment_name: Name of the experiment

    Returns:
        DataFrame with run comparisons
    """

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame()

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], output_format="pandas"
    )

    return runs
