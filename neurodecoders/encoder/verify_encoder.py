"""
Core encoder verification functionality for MLflow integration.

This module provides the EncoderVerifier class with essential analysis methods
used by the verification callback. All plotting functionality has been removed
as it's no longer needed for MLflow-based verification.
"""

import warnings
from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


class EncoderVerifier:
    """
    Core encoder verification and analysis tool for MLflow integration.

    This class provides essential analysis methods used by the verification
    callback. All plotting functionality has been removed as results are
    logged directly to MLflow.
    """

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.encoder = None
        self.images = None
        self.true_firing_rates = None
        self.predicted_firing_rates = None
        self.image_labels = None
        self.plots_dir = None

    def predict_firing_rates(self):
        """Generate predictions using the encoder model."""
        if self.encoder is None or self.images is None:
            raise ValueError(
                "Encoder and images must be set before prediction"
            )

        self.encoder.eval()
        predicted_rates = []

        with torch.no_grad():
            # Process in batches to avoid memory issues
            batch_size = 8
            for i in range(0, len(self.images), batch_size):
                batch_images = self.images[i : i + batch_size]

                if len(batch_images.shape) == 4:
                    batch_tensor = torch.tensor(
                        batch_images, dtype=torch.float32
                    ).to(self.device)
                else:
                    print(f"Unexpected image shape: {batch_images.shape}")
                    continue

                batch_predictions = self.encoder(batch_tensor)
                predicted_rates.append(batch_predictions.cpu().numpy())

        self.predicted_firing_rates = np.vstack(predicted_rates)
        print(f"Predictions shape: {self.predicted_firing_rates.shape}")

    def analyze_firing_rate_distributions(self) -> Dict[str, Any]:
        """Analyze firing rate distributions to detect issues."""
        print("\n=== FIRING RATE DISTRIBUTION ANALYSIS ===")

        # Basic statistics
        true_mean = np.mean(self.true_firing_rates, axis=0)
        true_std = np.std(self.true_firing_rates, axis=0)
        pred_mean = np.mean(self.predicted_firing_rates, axis=0)
        pred_std = np.std(self.predicted_firing_rates, axis=0)

        print(
            f"True firing rates - Mean: {np.mean(true_mean):.2f} ± "
            f"{np.mean(true_std):.2f}"
        )
        print(
            f"Predicted firing rates - Mean: {np.mean(pred_mean):.2f} ± "
            f"{np.mean(pred_std):.2f}"
        )

        # Check for constant predictions
        constant_neurons = []
        for i in range(self.predicted_firing_rates.shape[1]):
            if np.std(self.predicted_firing_rates[:, i]) < 0.1:
                constant_neurons.append(i)

        print(
            f"Neurons with constant predictions (< 0.1 std): "
            f"{len(constant_neurons)}/{self.predicted_firing_rates.shape[1]}"
        )
        if constant_neurons:
            print(f"Constant neuron indices: {constant_neurons[:10]}...")

        # Correlation analysis
        correlations = np.zeros(self.predicted_firing_rates.shape[1])
        for i in range(self.predicted_firing_rates.shape[1]):
            corr = np.corrcoef(
                self.true_firing_rates[:, i], self.predicted_firing_rates[:, i]
            )[0, 1]
            # Handle NaN values from constant arrays
            if np.isnan(corr):
                corr = 0.0  # Set correlation to 0 for constant arrays
            correlations[i] = corr
        print(
            f"Mean correlation between true and predicted: "
            f"{np.mean(correlations):.3f}"
        )
        print(f"Correlation std: {np.std(correlations):.3f}")
        print(
            f"Neurons with correlation > 0.5: "
            f"{np.sum(correlations > 0.5)}/{len(correlations)}"
        )

        return {
            "true_mean": true_mean,
            "pred_mean": pred_mean,
            "true_std": true_std,
            "pred_std": pred_std,
            "correlations": correlations,
            "constant_neurons": constant_neurons,
        }

    def test_image_classification_from_firing_rates(
        self,
    ) -> Optional[Dict[str, Any]]:
        """Test if predicted firing rates contain enough information for
        image classification."""
        print("\n=== IMAGE CLASSIFICATION FROM FIRING RATES ===")

        if self.image_labels is None:
            print("No image labels available. Skipping classification test.")
            return None

        X = self.predicted_firing_rates
        y = self.image_labels

        # Safety checks for tiny / imbalanced datasets
        unique_labels, counts = np.unique(y, return_counts=True)
        if len(unique_labels) < 2:
            print(
                "Not enough classes (need at least 2). "
                "Skipping classification test."
            )
            return {"skipped": True, "reason": "<2 classes"}
        if np.min(counts) < 2:
            print(
                "At least one class has fewer than 2 samples. "
                "Skipping classification test to avoid "
                "stratified split errors."
            )
            label_counts = {
                int(label): int(cnt)
                for label, cnt in zip(unique_labels, counts)
            }
            print(f"Class distribution: {label_counts}")
            return {"skipped": True, "reason": "class with <2 samples"}
        if len(y) < 20:
            print(
                f"Dataset very small (n={len(y)}). "
                "Skipping classification test for reliability."
            )
            return {"skipped": True, "reason": "dataset too small"}

        # Split data with stratification (safe now)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Test multiple classifiers
        classifiers = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            "SVM (RBF)": SVC(kernel="rbf", random_state=42),
            "SVM (Linear)": SVC(kernel="linear", random_state=42),
            "MLP (2 layers)": MLPClassifier(
                hidden_layer_sizes=(100, 50), max_iter=500, random_state=42
            ),
            "MLP (3 layers)": MLPClassifier(
                hidden_layer_sizes=(200, 100, 50),
                max_iter=500,
                random_state=42,
            ),
        }

        results = {}

        print("Testing different classifiers on PREDICTED firing rates:")
        print("-" * 60)

        for name, clf in classifiers.items():
            print(f"\nTraining {name}...")

            # For SVM and MLP, we need to scale the features
            if name.startswith("SVM") or name.startswith("MLP"):
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                clf.fit(X_train_scaled, y_train)
                y_pred = clf.predict(X_test_scaled)
            else:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            results[f"{name}_pred"] = accuracy
            print(f"{name} accuracy: {accuracy:.3f}")

        print("\n" + "=" * 60)
        print("Testing different classifiers on TRUE firing rates:")
        print("-" * 60)

        # Compare with true firing rates
        X_true = self.true_firing_rates
        X_true_train, X_true_test, y_train_true, y_test_true = (
            train_test_split(
                X_true, y, test_size=0.3, random_state=42, stratify=y
            )
        )

        for name, clf in classifiers.items():
            print(f"\nTraining {name} on TRUE firing rates...")

            # For SVM and MLP, we need to scale the features
            if name.startswith("SVM") or name.startswith("MLP"):
                scaler = StandardScaler()
                X_true_train_scaled = scaler.fit_transform(X_true_train)
                X_true_test_scaled = scaler.transform(X_true_test)

                clf.fit(X_true_train_scaled, y_train_true)
                y_pred_true = clf.predict(X_true_test_scaled)
            else:
                clf.fit(X_true_train, y_train_true)
                y_pred_true = clf.predict(X_true_test)

            accuracy_true = accuracy_score(y_test_true, y_pred_true)
            results[f"{name}_true"] = accuracy_true
            print(f"{name} accuracy (TRUE): {accuracy_true:.3f}")

        # Find best classifier for each type
        pred_accuracies = {
            k: v for k, v in results.items() if k.endswith("_pred")
        }
        true_accuracies = {
            k: v for k, v in results.items() if k.endswith("_true")
        }

        best_pred_classifier = max(pred_accuracies.items(), key=lambda x: x[1])
        best_true_classifier = max(true_accuracies.items(), key=lambda x: x[1])

        print("\n" + "=" * 60)
        print("SUMMARY:")
        print("-" * 60)
        print(
            f"Best classifier for PREDICTED firing rates: "
            f"{best_pred_classifier[0].replace('_pred', '')} "
            f"({best_pred_classifier[1]:.3f})"
        )
        print(
            f"Best classifier for TRUE firing rates: "
            f"{best_true_classifier[0].replace('_true', '')} "
            f"({best_true_classifier[1]:.3f})"
        )

        if best_true_classifier[1] > 0:
            degradation = (
                (best_true_classifier[1] - best_pred_classifier[1])
                / best_true_classifier[1]
            ) * 100
            print(f"Performance degradation: {degradation:.1f}%")

        return {"results": results}
