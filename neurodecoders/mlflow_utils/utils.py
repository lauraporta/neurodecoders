"""
Utility functions for MLflow operations in neurodecoders.

This module provides common utility functions for MLflow operations
that can be shared between encoder and decoder training.
"""

import os
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import pandas as pd
from pytorch_lightning import LightningModule


def setup_mlflow_experiment(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    artifact_location: Optional[str] = None,
) -> None:
    """
    Set up MLflow experiment with proper error handling.

    Args:
        experiment_name: Name of the MLflow experiment
        tracking_uri: MLflow tracking server URI (optional)
        artifact_location: Location for storing artifacts (optional)
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Set up experiment - handle deleted experiments gracefully
    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f"Warning: Could not set experiment '{experiment_name}': {e}")
        print("Creating new experiment...")
        try:
            mlflow.create_experiment(
                experiment_name, artifact_location=artifact_location
            )
            mlflow.set_experiment(experiment_name)
        except Exception as e2:
            print(f"Error creating experiment: {e2}")
            # Fall back to a non-default experiment name
            fallback = f"{experiment_name}_fallback"
            try:
                mlflow.create_experiment(
                    fallback, artifact_location=artifact_location
                )
            except Exception:
                pass
            mlflow.set_experiment(fallback)


def log_training_config(config: Dict[str, Any]) -> None:
    """
    Log training configuration to MLflow.

    Args:
        config: Training configuration dictionary
    """
    # Filter out None values and non-serializable objects
    loggable_config = {}
    for key, value in config.items():
        if value is not None and isinstance(value, (str, int, float, bool)):
            loggable_config[key] = value
        elif value is not None:
            # Convert other types to string
            loggable_config[key] = str(value)

    mlflow.log_params(loggable_config)


def log_model_artifacts(
    model: LightningModule,
    model_name: str = "model",
    model_type: str = "model",
    dataset_info: Optional[Dict[str, Any]] = None,
    training_info: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log model and related artifacts to MLflow.

    Args:
        model: The model to log
        model_name: Name for the model artifact
        model_type: Type of model (e.g., 'encoder', 'decoder')
        dataset_info: Information about the dataset used
        training_info: Information about the training process
    """
    # Log the model
    mlflow.pytorch.log_model(model, artifact_path=model_name)

    # Log metadata if provided
    if dataset_info or training_info:
        metadata: Dict[str, Any] = {
            "model_type": model_type,
            "model_name": model_name,
        }

        if dataset_info:
            metadata["dataset_info"] = dataset_info
        if training_info:
            metadata["training_info"] = training_info

        # Save metadata to temporary file
        metadata_path = "model_metadata.json"
        import json

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Log as artifact
        mlflow.log_artifact(metadata_path)

        # Clean up
        os.remove(metadata_path)


def log_training_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    prefix: str = "",
) -> None:
    """
    Log training metrics to MLflow.

    Args:
        metrics: Dictionary of metrics to log
        step: Step number for the metrics
        prefix: Prefix to add to metric names
    """
    if prefix:
        prefixed_metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    else:
        prefixed_metrics = metrics

    mlflow.log_metrics(prefixed_metrics, step=step)


def log_encoder_verification_metrics(
    verification_results: Dict[str, Any],
    step: Optional[int] = None,
) -> None:
    """
    Log encoder verification metrics to MLflow.

    This function extracts and logs all the verification metrics that were
    available in the original encoder implementation.

    Args:
        verification_results: Results from encoder verification analysis
        step: Step number for the metrics
    """
    metrics = {}

    # Extract key metrics from firing rate analysis
    if "firing_rates" in verification_results:
        firing_results = verification_results["firing_rates"]
        if "correlations" in firing_results:
            correlations = firing_results["correlations"]
            metrics["test_correlation"] = float(np.mean(correlations))
            metrics["test_correlation_std"] = float(np.std(correlations))

        # Add linear regression analysis for mean firing rates scatter plot
        if "true_mean" in firing_results and "pred_mean" in firing_results:
            true_mean = firing_results["true_mean"]
            pred_mean = firing_results["pred_mean"]

            # Linear regression for mean firing rates
            slope_mean, r2_mean = _calculate_linear_regression(
                true_mean, pred_mean
            )
            metrics["mean_firing_rate_slope"] = float(slope_mean)
            metrics["mean_firing_rate_r2"] = float(r2_mean)

        # Add linear regression analysis for std firing rates scatter plot
        if "true_std" in firing_results and "pred_std" in firing_results:
            true_std = firing_results["true_std"]
            pred_std = firing_results["pred_std"]

            # Linear regression for std firing rates
            slope_std, r2_std = _calculate_linear_regression(
                true_std, pred_std
            )
            metrics["std_firing_rate_slope"] = float(slope_std)
            metrics["std_firing_rate_r2"] = float(r2_std)

    # Extract classification metrics
    if "classification" in verification_results:
        class_results = verification_results["classification"]
        if "results" in class_results:
            results = class_results["results"]

            # Find best classifier accuracy for predicted firing rates
            pred_accuracies = {
                k: v for k, v in results.items() if k.endswith("_pred")
            }
            if pred_accuracies:
                best_pred_accuracy = max(pred_accuracies.values())
                metrics["best_classifier_accuracy_pred"] = float(
                    best_pred_accuracy
                )

            # Find best classifier accuracy for true firing rates
            true_accuracies = {
                k: v for k, v in results.items() if k.endswith("_true")
            }
            if true_accuracies:
                best_true_accuracy = max(true_accuracies.values())
                metrics["best_classifier_accuracy_true"] = float(
                    best_true_accuracy
                )

            # Calculate performance improvement
            if (
                best_true_accuracy > 0
                and "best_classifier_accuracy_pred" in metrics
            ):
                improvement = (
                    (best_pred_accuracy - best_true_accuracy)
                    / best_true_accuracy
                ) * 100
                metrics["classifier_performance_improvement"] = float(
                    improvement
                )

    # Log metrics to MLflow
    if metrics:
        mlflow.log_metrics(metrics, step=step)
        print(f"Logged {len(metrics)} verification metrics to MLflow")


def log_cross_validation_metrics(
    cv_results: Dict[str, Any],
    step: Optional[int] = None,
) -> None:
    """
    Log cross-validation metrics to MLflow.

    Args:
        cv_results: Cross-validation results dictionary
        step: Step number for the metrics
    """
    metrics = {}

    # Extract CV metrics
    if "train_losses" in cv_results:
        train_losses = cv_results["train_losses"]
        avg_train_loss = np.mean(train_losses)
        std_train_loss = np.std(train_losses)
        metrics["cv_avg_train_loss"] = float(avg_train_loss)
        metrics["cv_std_train_loss"] = float(std_train_loss)

    if "val_losses" in cv_results:
        val_losses = cv_results["val_losses"]
        avg_val_loss = np.mean(val_losses)
        std_val_loss = np.std(val_losses)
        metrics["cv_avg_val_loss"] = float(avg_val_loss)
        metrics["cv_std_val_loss"] = float(std_val_loss)

    if "test_losses" in cv_results:
        test_losses = cv_results["test_losses"]
        avg_test_loss = np.mean(test_losses)
        std_test_loss = np.std(test_losses)
        metrics["cv_avg_test_loss"] = float(avg_test_loss)
        metrics["cv_std_test_loss"] = float(std_test_loss)

    if "test_correlations" in cv_results:
        test_correlations = cv_results["test_correlations"]
        avg_correlation = np.mean(test_correlations)
        std_correlation = np.std(test_correlations)
        metrics["cv_avg_test_correlation"] = float(avg_correlation)
        metrics["cv_std_test_correlation"] = float(std_correlation)

    # Log metrics to MLflow
    if metrics:
        mlflow.log_metrics(metrics, step=step)
        print(f"Logged {len(metrics)} cross-validation metrics to MLflow")


def _calculate_linear_regression(x, y):
    """
    Calculate linear regression parameters for scatter plot analysis.

    Args:
        x: Independent variable (true values)
        y: Dependent variable (predicted values)

    Returns:
        tuple: (slope, r2_score)
    """
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        # Reshape for sklearn
        X = x.reshape(-1, 1)

        # Fit linear regression
        reg = LinearRegression()
        reg.fit(X, y)

        # Get slope and R²
        slope = reg.coef_[0]
        y_pred = reg.predict(X)
        r2 = r2_score(y, y_pred)

        return slope, r2

    except Exception as e:
        print(f"Error calculating linear regression: {e}")
        return 0.0, 0.0


def log_validation_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
) -> None:
    """
    Log validation metrics to MLflow.

    Args:
        metrics: Dictionary of validation metrics to log
        step: Step number for the metrics
    """
    log_training_metrics(metrics, step=step, prefix="val")


def log_test_metrics(
    metrics: Dict[str, float],
) -> None:
    """
    Log test metrics to MLflow.

    Args:
        metrics: Dictionary of test metrics to log
    """
    log_training_metrics(metrics, prefix="test")


def get_experiment_comparison(experiment_name: str = "neurodecoders"):
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


def log_artifacts_from_directory(
    local_dir: str,
    artifact_path: Optional[str] = None,
) -> None:
    """
    Log all artifacts from a local directory.

    Args:
        local_dir: Local directory containing artifacts
        artifact_path: Path within the run's artifact directory
    """
    if os.path.exists(local_dir):
        mlflow.log_artifacts(local_dir, artifact_path)
    else:
        print(f"Warning: Directory {local_dir} does not exist")


def log_single_artifact(
    local_path: str,
    artifact_path: Optional[str] = None,
) -> None:
    """
    Log a single artifact file.

    Args:
        local_path: Path to the local file
        artifact_path: Path within the run's artifact directory
    """
    if os.path.exists(local_path):
        mlflow.log_artifact(local_path, artifact_path)
    else:
        print(f"Warning: File {local_path} does not exist")


def log_dataset_input_and_params(data_module) -> None:
    """
    Log dataset metadata as MLflow Input and also as params.

    Mirrors the encoder's dataset logging so encoder/decoder can share it.
    """
    # Create metadata summary DataFrame
    metadata_summary = pd.DataFrame(
        [
            {
                "dataset_id": data_module.get_dataset_id(),
                "git_commit": data_module.git_commit,
                "git_branch": data_module.git_branch,
                "timestamp": data_module.timestamp,
                "images_shape": str(data_module.images.shape),
                "firing_rates_shape": str(data_module.firing_rates.shape),
                "labels_shape": (
                    str(data_module.labels.shape)
                    if hasattr(data_module, "labels")
                    and data_module.labels is not None
                    else "None"
                ),
                "total_size_mb": data_module.total_size_mb,
                "dataset_type": data_module.dataset_type,
                "sta_pattern": data_module.sta_pattern,
                "n_neurons": data_module.n_neurons,
                "n_images": data_module.n_images,
            }
        ]
    )

    # Derive a source info best-effort
    try:
        from neurodecoders.paths import get_path

        dataset_filename = (
            data_module.dataset_filename
            if hasattr(data_module, "dataset_filename")
            else "unknown"
        )
        source_info = (
            f"{get_path('workspace/datasets/synthetic')}/{dataset_filename}"
        )
    except Exception:
        source_info = "unknown"

    # Log the metadata dataset as MLflow input
    dataset_id = data_module.get_dataset_id()
    try:
        data_module.get_metadata_summary()
    except Exception:
        pass

    metadata_name = f"neural_data_{dataset_id}"
    summary_dataset = mlflow.data.from_pandas(
        metadata_summary, source=source_info, name=metadata_name
    )
    mlflow.log_input(summary_dataset, context="training_data")

    # Also log as parameters for backward compatibility
    try:
        dataset_params = data_module.get_mlflow_parameters()
        mlflow.log_params(dataset_params)
    except Exception:
        # Fall back to flattened subset
        images_attr = (
            data_module.images if hasattr(data_module, "images") else None
        )
        firing_rates_attr = (
            data_module.firing_rates
            if hasattr(data_module, "firing_rates")
            else None
        )

        fallback = {
            "dataset_id": dataset_id,
            "images_shape": (
                str(images_attr.shape) if images_attr is not None else "None"
            ),
            "firing_rates_shape": (
                str(firing_rates_attr.shape)
                if firing_rates_attr is not None
                else "None"
            ),
        }
        mlflow.log_params(fallback)
