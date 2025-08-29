"""
Verification callback for neural encoder training.

This module provides a PyTorch Lightning callback that automatically runs
encoder verification analysis after training completes, saving results to
both the workspace folder and MLflow.
"""

import datetime
import os
import sys
import traceback
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

# Add the encoder directory to the path for imports
sys.path.append(os.path.dirname(__file__))

from neurodecoders.paths import ensure_dir, get_path

from .verify_encoder import EncoderVerifier


class EncoderVerificationCallback(Callback):
    """
    PyTorch Lightning callback that runs encoder verification analysis
    after training completes.

    This callback automatically:
    1. Saves the trained model
    2. Runs comprehensive verification analysis
    3. Saves plots to workspace folder
    4. Logs results to MLflow
    """

    def __init__(
        self,
        data_module,
        save_model: bool = True,
        model_save_dir: str = get_path("workspace/models/encoders"),
        enable_mlflow_logging: bool = True,
    ):
        """
        Initialize the verification callback.

        Args:
            data_module: The data module used for training
            save_model: Whether to save the model before verification
            model_save_dir: Directory to save the model
            enable_mlflow_logging: Whether to log results to MLflow
        """
        super().__init__()
        self.data_module = data_module
        self.save_model = save_model
        self.model_save_dir = model_save_dir
        self.enable_mlflow_logging = enable_mlflow_logging

        # Create directories
        ensure_dir(self.model_save_dir)

        # Store training data for verification
        self.training_images = None
        self.training_firing_rates = None
        self.training_labels = None

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """Store training data for later verification."""
        # Extract training data from data module
        try:
            # Store the full training dataset for verification
            # This ensures we have all the data with correct labels
            self.training_images = self.data_module.images
            self.training_firing_rates = self.data_module.firing_rates

            # Ensure we have the data before proceeding
            assert self.training_images is not None
            assert self.training_firing_rates is not None

            # Store labels if available
            if (
                hasattr(self.data_module, "labels")
                and self.data_module.labels is not None
            ):
                self.training_labels = self.data_module.labels
                print(
                    f"Stored training data for verification: "
                    f"{self.training_images.shape} images, "
                    f"{self.training_firing_rates.shape[1]} neurons, "
                    f"{len(self.training_labels)} labels"
                )
            else:
                print(
                    f"Stored training data for verification: "
                    f"{self.training_images.shape} images, "
                    f"{self.training_firing_rates.shape[1]} neurons "
                    "(no labels available)"
                )

        except Exception as e:
            print(
                f"Warning: Could not store training data for verification: {e}"
            )

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Run verification analysis after training completes."""
        print("\n=== RUNNING ENCODER VERIFICATION ANALYSIS ===")

        # Save model if requested
        model_path = None
        if self.save_model:
            model_path = self._save_model(pl_module)

        # Run verification analysis
        verification_results = self._run_verification_analysis(
            pl_module, model_path
        )

        # Log results to MLflow if enabled
        if self.enable_mlflow_logging and verification_results:
            self._log_verification_to_mlflow(verification_results)

        print("=== VERIFICATION ANALYSIS COMPLETE ===")

    def _save_model(self, pl_module: pl.LightningModule) -> str:
        """Save the trained model."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = type(pl_module.model).__name__

        # Create model filename
        model_filename = f"{model_type.lower()}_{timestamp}.pth"
        model_path = os.path.join(self.model_save_dir, model_filename)

        # Save model state dict
        torch.save(pl_module.model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")

        return model_path

    def _run_verification_analysis(
        self, pl_module: pl.LightningModule, model_path: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Run the verification analysis."""
        try:
            # Create verifier
            verifier = EncoderVerifier()

            # Set the encoder model
            verifier.encoder = pl_module.model
            verifier.encoder.eval()

            # Use training data if available, otherwise create synthetic data
            if (
                self.training_images is not None
                and self.training_firing_rates is not None
            ):
                verifier.images = self.training_images
                verifier.true_firing_rates = self.training_firing_rates

                # Set labels if available
                if self.training_labels is not None:
                    verifier.image_labels = self.training_labels
                    print(
                        f"Using training data for verification: "
                        f"{verifier.images.shape} images, "
                        f"{verifier.true_firing_rates.shape[1]} neurons, "
                        f"{len(self.training_labels)} labels"
                    )
                else:
                    print(
                        f"Using training data for verification: "
                        f"{verifier.images.shape} images, "
                        f"{verifier.true_firing_rates.shape[1]} neurons "
                        "(no labels available)"
                    )
            else:
                # Load synthetic data from workspace for verification
                print(
                    "No training data available, loading synthetic data "
                    "from workspace for verification"
                )

                # Try to load synthetic data from workspace train split
                synthetic_dir = get_path("workspace/datasets/synthetic")
                train_dir = os.path.join(synthetic_dir, "train")
                if os.path.exists(train_dir):
                    available_files = [
                        f for f in os.listdir(train_dir) if f.endswith(".npz")
                    ]
                    if available_files:
                        # Use the most recent file
                        available_files.sort(reverse=True)
                        selected_file = available_files[0]
                        file_path = os.path.join(train_dir, selected_file)

                        try:
                            data = np.load(file_path)
                            if "images" in data and "responses" in data:
                                verifier.images = data["images"]
                                verifier.true_firing_rates = data["responses"]

                                # Load labels if available
                                if "labels" in data:
                                    verifier.image_labels = data["labels"]
                                    print(
                                        f"Loaded verification data from: "
                                        f"{selected_file} with labels"
                                    )
                                else:
                                    print(
                                        f"Loaded verification data from: "
                                        f"{selected_file} (no labels)"
                                    )
                            else:
                                raise ValueError(
                                    "Invalid synthetic data format"
                                )
                        except Exception as e:
                            print(
                                f"Error loading synthetic data "
                                "for verification: "
                                f"{e}"
                            )
                            raise RuntimeError(
                                "No verification data available"
                            )
                    else:
                        raise FileNotFoundError(
                            f"No synthetic data files found in {synthetic_dir}"
                        )
                else:
                    raise FileNotFoundError(
                        f"Synthetic data directory {synthetic_dir} not found"
                    )

            # Generate predictions
            verifier.predict_firing_rates()

            # Run analyses
            results: Dict[str, Any] = {}

            # Firing rate analysis
            print("Running firing rate analysis...")
            results["firing_rates"] = (
                verifier.analyze_firing_rate_distributions()
            )

            # Classification test (if labels are available)
            if (
                hasattr(verifier, "image_labels")
                and verifier.image_labels is not None
            ):
                print("Running classification analysis...")
                classification_results = (
                    verifier.test_image_classification_from_firing_rates()
                )
                if classification_results:
                    results["classification"] = classification_results

            # Store model path in results
            results["model_path"] = model_path

            print("Verification analysis complete.")

            return results

        except Exception as e:
            print(f"Error during verification analysis: {e}")
            traceback.print_exc()
            return None

    def _log_verification_to_mlflow(
        self, verification_results: Dict[str, Any]
    ):
        """Log verification results to MLflow."""
        try:
            # Log verification metrics
            metrics = {}

            # Extract key metrics from firing rate analysis
            if "firing_rates" in verification_results:
                firing_results = verification_results["firing_rates"]
                if "correlations" in firing_results:
                    correlations = firing_results["correlations"]
                    metrics["test_correlation"] = float(np.mean(correlations))
                    # Add correlation std as requested
                    metrics["test_correlation_std"] = float(
                        np.std(correlations)
                    )

                # Add linear regression analysis for mean firing
                # rates scatter plot
                if (
                    "true_mean" in firing_results
                    and "pred_mean" in firing_results
                ):
                    true_mean = firing_results["true_mean"]
                    pred_mean = firing_results["pred_mean"]

                    # Linear regression for mean firing rates
                    slope_mean, r2_mean = self._calculate_linear_regression(
                        true_mean, pred_mean
                    )
                    metrics["mean_firing_rate_slope"] = float(slope_mean)
                    metrics["mean_firing_rate_r2"] = float(r2_mean)

                # Add linear regression analysis for std firing
                # rates scatter plot
                if (
                    "true_std" in firing_results
                    and "pred_std" in firing_results
                ):
                    true_std = firing_results["true_std"]
                    pred_std = firing_results["pred_std"]

                    # Linear regression for std firing rates
                    slope_std, r2_std = self._calculate_linear_regression(
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

                    # Calculate performance degradation
                    if (
                        best_true_accuracy > 0
                        and "best_classifier_accuracy_pred" in metrics
                    ):
                        degradation = (
                            (best_true_accuracy - best_pred_accuracy)
                            / best_true_accuracy
                        ) * 100
                        metrics["classifier_performance_degradation"] = float(
                            degradation
                        )

            # Log metrics to MLflow
            if metrics:
                mlflow.log_metrics(metrics)
                print(f"Logged {len(metrics)} verification metrics to MLflow")

            # Log model path if available
            model_path = verification_results.get("model_path")
            if model_path and os.path.exists(model_path):
                mlflow.log_artifact(model_path, "trained_model")
                print(f"Logged trained model to MLflow: {model_path}")

        except Exception as e:
            print(f"Error logging verification results to MLflow: {e}")
            traceback.print_exc()

    def _calculate_linear_regression(self, x, y):
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
