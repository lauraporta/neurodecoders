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

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

# Add the encoder directory to the path for imports
sys.path.append(os.path.dirname(__file__))

from neurodecoders.mlflow_utils.utils import (
    create_all_firing_rates_scatterplot,
    create_firing_rate_scatterplot,
    log_encoder_verification_metrics,
    log_single_artifact,
)
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
            try:
                log_encoder_verification_metrics(verification_results)
                print(
                    "Logged verification metrics via shared MLflow utilities"
                )
            except Exception as e:
                print(f"Error logging verification metrics to MLflow: {e}")
                traceback.print_exc()

            # Log model artifact if available
            model_path = verification_results.get("model_path")
            if model_path and os.path.exists(model_path):
                try:
                    log_single_artifact(model_path, "trained_model")
                    print(f"Logged trained model to MLflow: {model_path}")
                except Exception as e:
                    print(f"Error logging trained model artifact: {e}")
                    traceback.print_exc()

            # Log firing rate plot artifact if available
            plot_path = verification_results.get("firing_rate_plot_path")
            if plot_path and os.path.exists(plot_path):
                try:
                    log_single_artifact(plot_path, "firing_rate_analysis_plot")
                    print(f"Logged firing rate plot to MLflow: {plot_path}")
                except Exception as e:
                    print(f"Error logging firing rate plot artifact: {e}")
                    traceback.print_exc()

            # Log all firing rates plot artifact if available
            all_rates_plot_path = verification_results.get("all_firing_rates_plot_path")
            if all_rates_plot_path and os.path.exists(all_rates_plot_path):
                try:
                    log_single_artifact(all_rates_plot_path, "all_firing_rates_plot")
                    print(f"Logged all firing rates plot to MLflow: {all_rates_plot_path}")
                except Exception as e:
                    print(f"Error logging all firing rates plot artifact: {e}")
                    traceback.print_exc()

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

            # Generate scatterplot visualization
            print("Generating firing rate scatterplot...")
            plot_path = self._generate_firing_rate_plot(
                results["firing_rates"], model_path
            )
            if plot_path:
                results["firing_rate_plot_path"] = plot_path

            # Generate all firing rates scatterplot
            print("Generating all firing rates scatterplot...")
            all_rates_plot_path = self._generate_all_firing_rates_plot(
                verifier, model_path
            )
            if all_rates_plot_path:
                results["all_firing_rates_plot_path"] = all_rates_plot_path

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

    def _generate_firing_rate_plot(
        self, firing_results: Dict[str, Any], model_path: Optional[str]
    ) -> Optional[str]:
        """Generate and save firing rate scatterplot visualization."""
        try:
            # Extract firing rate data
            true_mean = firing_results.get("true_mean")
            pred_mean = firing_results.get("pred_mean")
            true_std = firing_results.get("true_std")
            pred_std = firing_results.get("pred_std")

            if (
                true_mean is None
                or pred_mean is None
                or true_std is None
                or pred_std is None
            ):
                print("Missing firing rate data for plot generation")
                return None

            # Create plots directory
            plots_dir = get_path("workspace/plots")
            ensure_dir(plots_dir)

            # Generate timestamp for unique filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"firing_rate_analysis_{timestamp}.png"
            plot_path = os.path.join(plots_dir, plot_filename)

            # Create the scatterplot
            plot_path = create_firing_rate_scatterplot(
                true_mean=true_mean,
                pred_mean=pred_mean,
                true_std=true_std,
                pred_std=pred_std,
                save_path=plot_path,
                title=f"Firing Rate Analysis - {timestamp}",
            )

            if plot_path and os.path.exists(plot_path):
                print(f"Firing rate scatterplot saved to: {plot_path}")
                return plot_path
            else:
                print("Failed to generate firing rate scatterplot")
                return None

        except Exception as e:
            print(f"Error generating firing rate plot: {e}")
            traceback.print_exc()
            return None

    def _generate_all_firing_rates_plot(
        self, verifier, model_path: Optional[str]
    ) -> Optional[str]:
        """Generate and save all firing rates scatterplot visualization."""
        try:
            # Check if we have the required data
            if (
                not hasattr(verifier, "true_firing_rates")
                or not hasattr(verifier, "predicted_firing_rates")
                or verifier.true_firing_rates is None
                or verifier.predicted_firing_rates is None
            ):
                print("Missing firing rate arrays for all rates plot generation")
                return None

            # Create plots directory
            plots_dir = get_path("workspace/plots")
            ensure_dir(plots_dir)

            # Generate timestamp for unique filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"all_firing_rates_{timestamp}.png"
            plot_path = os.path.join(plots_dir, plot_filename)

            # Create the scatterplot
            plot_path = create_all_firing_rates_scatterplot(
                true_firing_rates=verifier.true_firing_rates,
                pred_firing_rates=verifier.predicted_firing_rates,
                save_path=plot_path,
                title=f"All Firing Rates - {timestamp}",
            )

            if plot_path and os.path.exists(plot_path):
                print(f"All firing rates scatterplot saved to: {plot_path}")
                return plot_path
            else:
                print("Failed to generate all firing rates scatterplot")
                return None

        except Exception as e:
            print(f"Error generating all firing rates plot: {e}")
            traceback.print_exc()
            return None
