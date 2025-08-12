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
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

    def analyze_neuron_responsiveness(self) -> Dict[str, Any]:
        """Analyze how responsive neurons are to different images."""
        print("\n=== NEURON RESPONSIVENESS ANALYSIS ===")

        # Calculate responsiveness (how much firing rate varies across images)
        true_responsiveness = np.std(self.true_firing_rates, axis=0)
        pred_responsiveness = np.std(self.predicted_firing_rates, axis=0)

        # Find neurons with high true responsiveness but low predicted
        # responsiveness
        high_true_low_pred = []
        for i in range(len(true_responsiveness)):
            if true_responsiveness[i] > np.percentile(
                true_responsiveness, 75
            ) and pred_responsiveness[i] < np.percentile(
                pred_responsiveness, 25
            ):
                high_true_low_pred.append(i)

        print(
            f"Neurons with high true responsiveness but low predicted "
            f"responsiveness: {len(high_true_low_pred)}"
        )

        return {
            "true_responsiveness": true_responsiveness,
            "pred_responsiveness": pred_responsiveness,
            "high_true_low_pred": high_true_low_pred,
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

        # Use predicted firing rates as features
        X = self.predicted_firing_rates
        y = self.image_labels

        # Split data
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

    def analyze_encoder_representations(self) -> Dict[str, Any]:
        """Analyze the learned representations using PCA."""
        print("\n=== ENCODER REPRESENTATION ANALYSIS ===")

        # Use predicted firing rates for analysis
        X = self.predicted_firing_rates

        # Calculate variance statistics
        variances = np.var(X, axis=0)
        mean_variance = np.mean(variances)
        std_variance = np.std(variances)
        min_variance = np.min(variances)
        max_variance = np.max(variances)

        print("Data variance statistics:")
        print(f"  Mean variance across neurons: {mean_variance:.6f}")
        print(f"  Std variance across neurons: {std_variance:.6f}")
        print(f"  Min variance: {min_variance:.6f}")
        print(f"  Max variance: {max_variance:.6f}")

        # PCA analysis
        pca = PCA()
        pca.fit(X)
        explained_variance_ratio = pca.explained_variance_ratio_

        print("PCA explained variance:")
        print(
            f"  First 5 components: {np.sum(explained_variance_ratio[:5]):.6f}"
        )
        print(
            f"  First 10 components: "
            f"{np.sum(explained_variance_ratio[:10]):.6f}"
        )
        print(
            f"  First 20 components: "
            f"{np.sum(explained_variance_ratio[:20]):.6f}"
        )

        # Find components needed for different variance thresholds
        cumulative_variance = np.cumsum(explained_variance_ratio)
        thresholds = [0.5, 0.8, 0.9, 0.95]
        components_needed = {}

        for threshold in thresholds:
            components_needed[threshold] = (
                np.argmax(cumulative_variance >= threshold) + 1
            )
            print(
                f"  Components needed for {threshold * 100:.0f}% variance: "
                f"{components_needed[threshold]}"
            )

        print("\nFirst 10 components explained variance:")
        for i in range(min(10, len(explained_variance_ratio))):
            print(f"  Component {i + 1}: {explained_variance_ratio[i]:.6f}")

        return {
            "explained_variance_ratio": explained_variance_ratio,
            "components_needed": components_needed,
            "mean_variance": mean_variance,
            "std_variance": std_variance,
        }

    def analyze_feature_scaling_and_separability(self) -> Dict[str, Any]:
        """Analyze feature scaling and linear separability."""
        print("\n=== FEATURE SCALING AND LINEAR SEPARABILITY ANALYSIS ===")

        if self.image_labels is None:
            print("No image labels available. Skipping analysis.")
            return {}

        # Use predicted firing rates for analysis
        X = self.predicted_firing_rates
        y = self.image_labels

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        print("Testing different scaling methods with Linear SVM:")
        print("-" * 50)

        # Test different scaling methods
        scaling_methods = {
            "No Scaling": None,
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
        }

        scaling_results = {}
        for name, scaler in scaling_methods.items():
            if scaler is None:
                X_train_scaled = X_train
                X_test_scaled = X_test
            else:
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

            svm = SVC(kernel="linear", random_state=42)
            svm.fit(X_train_scaled, y_train)
            y_pred = svm.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            scaling_results[name] = accuracy
            print(f"{name}: {accuracy:.3f}")

        # Feature distribution analysis
        print("\nFeature distribution analysis:")
        print("-" * 50)

        pred_mean = np.mean(X, axis=0)
        pred_std = np.std(X, axis=0)
        true_mean = np.mean(self.true_firing_rates, axis=0)
        true_std = np.std(self.true_firing_rates, axis=0)

        print("Predicted firing rates:")
        print(f"  Mean: {np.mean(pred_mean):.2f} ± {np.std(pred_mean):.2f}")
        print(f"  Std: {np.mean(pred_std):.2f} ± {np.std(pred_std):.2f}")
        print(f"  Range: [{np.min(pred_mean):.2f}, {np.max(pred_mean):.2f}]")
        print(
            f"  Coefficient of variation: {np.mean(pred_std / pred_mean):.3f}"
        )

        print("\nTrue firing rates:")
        print(f"  Mean: {np.mean(true_mean):.2f} ± {np.std(true_mean):.2f}")
        print(f"  Std: {np.mean(true_std):.2f} ± {np.std(true_std):.2f}")
        print(f"  Range: [{np.min(true_mean):.2f}, {np.max(true_mean):.2f}]")
        print(
            f"  Coefficient of variation: {np.mean(true_std / true_mean):.3f}"
        )

        # Linear separability analysis using Fisher's discriminant
        print("\nLinear separability analysis:")
        print("-" * 50)

        # Calculate Fisher's discriminant ratio for predicted vs true
        def fisher_discriminant_ratio(X, y):
            unique_labels = np.unique(y)
            if len(unique_labels) < 2:
                return 0.0

            # Calculate between-class and within-class scatter
            overall_mean = np.mean(X, axis=0)
            between_class_scatter = 0
            within_class_scatter = 0

            for label in unique_labels:
                class_samples = X[y == label]
                class_mean = np.mean(class_samples, axis=0)
                class_size = len(class_samples)

                # Between-class scatter
                diff = class_mean - overall_mean
                between_class_scatter += class_size * np.outer(diff, diff)

                # Within-class scatter
                for sample in class_samples:
                    diff = sample - class_mean
                    within_class_scatter += np.outer(diff, diff)

            # Calculate Fisher's discriminant ratio
            if np.linalg.det(within_class_scatter) > 1e-10:
                fisher_ratio = np.trace(
                    np.linalg.inv(within_class_scatter) @ between_class_scatter
                )
                return fisher_ratio
            else:
                return 0.0

        pred_fisher = fisher_discriminant_ratio(X, y)
        true_fisher = fisher_discriminant_ratio(self.true_firing_rates, y)

        print("Fisher's discriminant ratio (higher = better separability):")
        print(f"  Predicted firing rates: {pred_fisher:.3f}")
        print(f"  True firing rates: {true_fisher:.3f}")
        if true_fisher > 0:
            print(f"  Ratio (pred/true): {pred_fisher / true_fisher:.3f}")
        else:
            print("  Ratio (pred/true): 0.000")

        # Support vector analysis
        print("\nSupport vector analysis:")
        print("-" * 50)

        svm = SVC(kernel="linear", random_state=42)
        svm.fit(X_train, y_train)

        n_support_vectors = len(svm.support_vectors_)
        total_samples = len(X_train)
        support_vector_ratio = n_support_vectors / total_samples
        margin_size = (
            1.0 / np.sqrt(np.sum(svm.coef_**2)) if len(svm.coef_) > 0 else 0
        )

        print(f"Number of support vectors: {n_support_vectors}")
        print(f"Total training samples: {total_samples}")
        print(f"Support vector ratio: {support_vector_ratio:.3f}")
        print(f"Margin size: {margin_size:.6f}")

        # Feature importance analysis
        if len(svm.coef_) > 0:
            feature_importance = np.abs(svm.coef_[0])
            top_features = np.argsort(feature_importance)[::-1][:10]

            print("\nTop 10 most important features (neurons) for linear SVM:")
            for i, feature_idx in enumerate(top_features):
                weight = svm.coef_[0][feature_idx]
                print(
                    f"  {i + 1}. Neuron {feature_idx}: weight = {weight:.4f}"
                )

        return {
            "scaling_results": scaling_results,
            "pred_fisher_ratio": pred_fisher,
            "true_fisher_ratio": true_fisher,
            "support_vector_ratio": support_vector_ratio,
            "margin_size": margin_size,
        }

    def suggest_improvements(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate improvement suggestions based on analysis results."""
        print("\n=== SUGGESTED IMPROVEMENTS ===")

        suggestions = []

        # Check firing rate analysis
        if "firing_rates" in results:
            firing_results = results["firing_rates"]
            if "correlations" in firing_results:
                mean_corr = np.mean(firing_results["correlations"])
                if mean_corr < 0.3:
                    suggestions.append(
                        "Low correlation between true and predicted "
                        "firing rates"
                    )
                    suggestions.append("Consider increasing model capacity")
                    suggestions.append("Try different loss functions")
                    suggestions.append("Check data preprocessing")

                if len(firing_results.get("constant_neurons", [])) > 0:
                    suggestions.append(
                        f"Found {len(firing_results['constant_neurons'])} "
                        "neurons with constant predictions"
                    )
                    suggestions.append("Check for vanishing gradients")
                    suggestions.append("Try different initialization")

        # Check classification results
        if "classification" in results:
            class_results = results["classification"]
            if "results" in class_results:
                results_dict = class_results["results"]
                pred_accuracies = {
                    k: v
                    for k, v in results_dict.items()
                    if k.endswith("_pred")
                }
                if pred_accuracies:
                    best_pred_acc = max(pred_accuracies.values())
                    if best_pred_acc < 0.5:
                        suggestions.append(
                            "Poor image classification from predicted "
                            "firing rates"
                        )
                        suggestions.append(
                            "The encoder is not learning meaningful "
                            "representations"
                        )
                        suggestions.append(
                            "Consider using contrastive learning"
                        )
                        suggestions.append(
                            "Add reconstruction loss as auxiliary task"
                        )
                        suggestions.append(
                            "Use pre-trained vision encoders and fine-tune"
                        )
                        suggestions.append(
                            "Implement multi-task learning with image "
                            "classification"
                        )

        # Check representation analysis
        if "representations" in results:
            rep_results = results["representations"]
            if "components_needed" in rep_results:
                comp_needed = rep_results["components_needed"]
                if comp_needed.get(0.8, 0) > 5:
                    suggestions.append(
                        "High-dimensional representations detected"
                    )
                    suggestions.append("Consider dimensionality reduction")
                    suggestions.append("Try regularization techniques")

        # Print suggestions
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                print(f"{i}. {suggestion}")
        else:
            print(
                "No specific issues detected. Model appears to be "
                "performing well."
            )

        return {"suggestions": suggestions}
