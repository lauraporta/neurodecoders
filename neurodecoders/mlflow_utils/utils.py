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

    # Resolve or create the experiment in a race-safe way
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        try:
            experiment_id = mlflow.create_experiment(
                experiment_name, artifact_location=artifact_location
            )
        except Exception as e:
            # Another process may have created it; fetch again
            fetched = mlflow.get_experiment_by_name(experiment_name)
            if fetched is None:
                # As a last resort, fall back to a deterministic name
                fallback = f"{experiment_name}_fallback"
                try:
                    experiment_id = mlflow.create_experiment(
                        fallback, artifact_location=artifact_location
                    )
                except Exception:
                    # If even fallback fails, set to Default
                    mlflow.set_experiment("Default")
                    return
            else:
                experiment_id = fetched.experiment_id
    else:
        experiment_id = exp.experiment_id

    # Set by experiment_id to avoid name/ID mismatches
    mlflow.set_experiment(experiment_id=experiment_id)


def log_training_config(config: Dict[str, Any]) -> None:
    """
    Log training configuration to MLflow.

    Args:
        config: Dictionary of training configuration parameters
    """
    # Convert complex objects to string representations
    sanitized_config: Dict[str, Any] = {}
    for k, v in config.items():
        try:
            _ = hash(v)
            sanitized_config[k] = v
        except Exception:
            sanitized_config[k] = str(v)

    # Log the sanitized configuration
    mlflow.log_params(sanitized_config)


def log_training_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Log training metrics to MLflow.

    Args:
        metrics: Dictionary of metric name to value
        step: Optional step number for logging
    """
    mlflow.log_metrics(metrics, step=step)


def log_dataset_input_and_params(
    dataset_name: str,
    n_images: int,
    n_neurons: int,
    additional_params: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log dataset-related input parameters to MLflow.

    Args:
        dataset_name: Name of the dataset
        n_images: Number of images used
        n_neurons: Number of neurons
        additional_params: Optional dictionary of additional parameters
    """
    params = {
        "dataset_name": dataset_name,
        "n_images": int(n_images),
        "n_neurons": int(n_neurons),
    }
    if additional_params:
        # Ensure values are JSON/param safe
        for k, v in additional_params.items():
            try:
                _ = hash(v)
                params[k] = v
            except Exception:
                params[k] = str(v)
    mlflow.log_params(params)


def log_model_artifacts(model: LightningModule, artifact_subdir: str = "model") -> None:
    """
    Log model artifacts to MLflow.

    Args:
        model: The model to log
        artifact_subdir: Subdirectory name for the logged model
    """
    # Placeholder for richer artifact logging; keep minimal to avoid import-time
    # heavy dependencies outside of mlflow itself.
    try:
        import mlflow.pytorch

        mlflow.pytorch.log_model(model, artifact_path=artifact_subdir)
    except Exception as e:
        print(f"Warning: Could not log model artifacts: {e}")


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


def create_firing_rate_scatterplot(
    true_mean: np.ndarray,
    pred_mean: np.ndarray,
    true_std: np.ndarray,
    pred_std: np.ndarray,
    save_path: str,
    title: str = "Firing Rate Analysis: Predicted vs True",
) -> str:
    """
    Create a scatterplot visualization of mean firing rates with standard
    deviation and fit line.

    Args:
        true_mean: True mean firing rates
        pred_mean: Predicted mean firing rates
        true_std: True standard deviation of firing rates
        pred_std: Predicted standard deviation of firing rates
        save_path: Path to save the plot
        title: Title for the plot

    Returns:
        str: Path to the saved plot file
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Mean firing rates
        ax1.scatter(true_mean, pred_mean, alpha=0.6, s=30)

        # Add fit line for mean firing rates
        if len(true_mean) > 1:
            reg_mean = LinearRegression()
            X_mean = true_mean.reshape(-1, 1)
            reg_mean.fit(X_mean, pred_mean)
            pred_line_mean = reg_mean.predict(X_mean)
            r2_mean = r2_score(pred_mean, pred_line_mean)

            # Plot fit line
            ax1.plot(
                true_mean,
                pred_line_mean,
                "r-",
                label=f"Fit (R²={r2_mean:.3f})",
            )
            ax1.legend()

        ax1.set_xlabel("True Mean Firing Rate")
        ax1.set_ylabel("Predicted Mean Firing Rate")
        ax1.set_title("Mean Firing Rates")
        ax1.grid(True, alpha=0.3)

        # Add diagonal line for perfect prediction
        min_val = min(true_mean.min(), pred_mean.min())
        max_val = max(true_mean.max(), pred_mean.max())
        ax1.plot(
            [min_val, max_val],
            [min_val, max_val],
            "k--",
            alpha=0.5,
            label="Perfect Prediction",
        )
        ax1.legend()

        # Plot 2: Standard deviation of firing rates
        ax2.scatter(true_std, pred_std, alpha=0.6, s=30, color="orange")

        # Add fit line for std firing rates
        if len(true_std) > 1:
            reg_std = LinearRegression()
            X_std = true_std.reshape(-1, 1)
            reg_std.fit(X_std, pred_std)
            pred_line_std = reg_std.predict(X_std)
            r2_std = r2_score(pred_std, pred_line_std)

            # Plot fit line
            ax2.plot(
                true_std, pred_line_std, "r-", label=f"Fit (R²={r2_std:.3f})"
            )
            ax2.legend()

        ax2.set_xlabel("True Std Firing Rate")
        ax2.set_ylabel("Predicted Std Firing Rate")
        ax2.set_title("Standard Deviation of Firing Rates")
        ax2.grid(True, alpha=0.3)

        # Add diagonal line for perfect prediction
        min_std = min(true_std.min(), pred_std.min())
        max_std = max(true_std.max(), pred_std.max())
        ax2.plot(
            [min_std, max_std],
            [min_std, max_std],
            "k--",
            alpha=0.5,
            label="Perfect Prediction",
        )
        ax2.legend()

        # Overall title
        fig.suptitle(title, fontsize=14, fontweight="bold")

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return save_path

    except Exception as e:
        print(f"Error creating firing rate scatterplot: {e}")
        return ""
