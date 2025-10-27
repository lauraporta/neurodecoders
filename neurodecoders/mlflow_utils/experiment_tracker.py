"""
Enhanced MLflow experiment tracker for neurodecoders.

This module provides a unified interface for tracking experiments,
logging hyperparameters, metrics, and model artifacts for both
encoder and decoder training.
"""

import json
import os
from typing import Any, Dict, Optional, Union

import mlflow
import mlflow.pytorch
import torch.nn as nn
from pytorch_lightning import LightningModule


class MLflowExperimentTracker:
    """
    Enhanced MLflow experiment tracker for neurodecoders training.

    This class provides a unified interface for tracking experiments,
    logging hyperparameters, metrics, and model artifacts for both
    encoder and decoder training.
    """

    def __init__(
        self,
        experiment_name: str = "neurodecoders",
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

        # Set tracking URI if provided, otherwise use config
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Load from config (supports both database and file system)
            try:
                from neurodecoders.config import get_mlflow_tracking_uri
                config_uri = get_mlflow_tracking_uri()
                if config_uri:
                    mlflow.set_tracking_uri(config_uri)
            except Exception as e:
                print(f"Warning: Could not load tracking URI from config: {e}")

        # Resolve or create the experiment in a race-safe way
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            try:
                experiment_id = mlflow.create_experiment(
                    experiment_name, artifact_location=artifact_location
                )
            except Exception:
                # Another process may have created it; fetch again
                fetched = mlflow.get_experiment_by_name(experiment_name)
                if fetched is None:
                    # Fall back to Default to avoid crashes
                    mlflow.set_experiment("Default")
                    return
                else:
                    experiment_id = fetched.experiment_id
        else:
            experiment_id = exp.experiment_id

        mlflow.set_experiment(experiment_id=experiment_id)

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

    def log_model(
        self,
        model: Union[LightningModule, nn.Module],
        model_name: str = "model",
        registered_model_name: Optional[str] = None,
    ):
        """
        Log a PyTorch model (Lightning or regular).

        Args:
            model: The model to log (LightningModule or nn.Module)
            model_name: Name for the model artifact
            registered_model_name: Name for model registry (optional)
        """
        if isinstance(model, LightningModule):
            mlflow.pytorch.log_model(
                model,
                artifact_path=model_name,
                registered_model_name=registered_model_name,
            )
        else:
            # For regular nn.Module
            mlflow.pytorch.log_model(
                model,
                artifact_path=model_name,
                registered_model_name=registered_model_name,
            )

    def log_model_metadata(
        self,
        model_path: str,
        model_type: str,
        dataset_info: Dict[str, Any],
        training_info: Dict[str, Any],
    ):
        """
        Log metadata about the trained model and training context.
        """
        metadata = {
            "model_path": model_path,
            "model_type": model_type,
            "dataset_info": dataset_info,
            "training_info": training_info,
        }
        temp_path = "model_metadata.json"
        with open(temp_path, "w") as f:
            json.dump(metadata, f, indent=2)
        mlflow.log_artifact(temp_path)
        os.remove(temp_path)
