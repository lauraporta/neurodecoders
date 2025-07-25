import glob
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from neurodecoders.encoder.encoder import SimpleEncoder

warnings.filterwarnings("ignore")


class EncoderVerifier:
    """
    Comprehensive encoder verification and analysis tool.
    Diagnoses encoder issues and evaluates its quality for image classification.
    """

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.encoder = None
        self.images = None
        self.true_firing_rates = None
        self.predicted_firing_rates = None
        self.image_labels = None
        self.plots_dir = None

    def setup_plots_directory(self, model_path):
        """Create a dedicated directory for saving verification plots"""
        import datetime
        from pathlib import Path
        
        # Extract model name and timestamp for folder naming
        model_name = Path(model_path).stem
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create plots directory
        self.plots_dir = f"verification_plots/{model_name}_{timestamp}"
        os.makedirs(self.plots_dir, exist_ok=True)
        
        print(f"Plots will be saved to: {self.plots_dir}")
        return self.plots_dir

    def save_plot(self, filename, dpi=300, bbox_inches="tight"):
        """Save plot to the plots directory"""
        if self.plots_dir is None:
            self.plots_dir = "verification_plots/default"
            os.makedirs(self.plots_dir, exist_ok=True)
        
        full_path = os.path.join(self.plots_dir, filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches=bbox_inches)
        plt.close()  # Close the figure to free memory
        print(f"Saved plot: {full_path}")

    def load_encoder_and_data(self, model_path, data_path=None):
        """Load encoder model and corresponding data"""
        print(f"Loading encoder from: {model_path}")
        
        # Setup plots directory
        self.setup_plots_directory(model_path)

        # Load encoder
        try:
            state_dict = torch.load(model_path, map_location=self.device)
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        # Determine encoder type from model path
        model_name = os.path.basename(model_path)
        if "resnet" in model_name.lower():
            encoder_type = "resnet"
            print("Detected ResNet encoder")
        else:
            encoder_type = "simple"
            print("Detected Simple encoder")

        # Determine output neurons based on encoder type
        if encoder_type == "resnet":
            # For ResNet, look for firing_head layers
            if "firing_head.6.weight" in state_dict:
                out_neurons = state_dict["firing_head.6.weight"].shape[0]
            elif "model.firing_head.6.weight" in state_dict:
                out_neurons = state_dict["model.firing_head.6.weight"].shape[0]
            else:
                # Find the last firing_head layer
                firing_head_keys = [
                    k
                    for k in state_dict.keys()
                    if "firing_head" in k and "weight" in k
                ]
                if not firing_head_keys:
                    print("No firing_head layers found in ResNet model")
                    return
                last_firing_head_key = sorted(firing_head_keys)[-1]
                out_neurons = state_dict[last_firing_head_key].shape[0]

            # Import and create ResNet encoder
            from neurodecoders.encoder.resnet_encoder import ResNetEncoder

            self.encoder = ResNetEncoder(out_neurons, resnet_type="resnet18")

        else:  # Simple encoder
            # Determine output neurons for simple encoder
            if "model.fc.6.weight" in state_dict:
                out_neurons = state_dict["model.fc.6.weight"].shape[0]
            elif "fc.6.weight" in state_dict:
                out_neurons = state_dict["fc.6.weight"].shape[0]
            else:
                # Find the last fc layer
                fc_keys = [
                    k for k in state_dict.keys() if "fc" in k and "weight" in k
                ]
                if not fc_keys:
                    print("No fc layers found in model")
                    return
                last_fc_key = sorted(fc_keys)[-1]
                out_neurons = state_dict[last_fc_key].shape[0]

            self.encoder = SimpleEncoder(out_neurons)

        # Handle nested model structure for both encoder types
        if any(k.startswith("model.") for k in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("model."):
                    new_key = key[6:]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict

        try:
            self.encoder.load_state_dict(state_dict)
            self.encoder.to(self.device)
            self.encoder.eval()
        except Exception as e:
            print(f"Error loading state dict: {e}")
            return

        print(f"Encoder loaded with {out_neurons} output neurons")

        # Load data
        if data_path is None:
            # Try to find corresponding data file
            model_name = os.path.basename(model_path).replace(".pth", "")
            print(f"Looking for data file matching model: {model_name}")

            # Try different patterns
            patterns = [
                f"data/synthdata_dataset-*{model_name.split('_datetime-')[0]}*.npz",
                "data/synthdata_dataset-*.npz",
            ]

            data_path = None
            for pattern in patterns:
                data_files = glob.glob(pattern)
                if data_files:
                    data_path = max(data_files, key=os.path.getctime)
                    print(f"Found data file: {data_path}")
                    break

            if data_path is None:
                print("No matching data file found. Please provide data_path.")
                print("Available data files:")
                for f in glob.glob("data/synthdata_dataset-*.npz"):
                    print(f"  {f}")
                return

        print(f"Loading data from: {data_path}")
        try:
            data = np.load(data_path)
            print(f"Data keys: {list(data.keys())}")

            if "images" in data:
                self.images = data["images"]
            else:
                print("No 'images' key found in data file")
                return

            if "responses" in data:
                self.true_firing_rates = data["responses"]
            else:
                print("No 'responses' key found in data file")
                return

            # Extract image labels if available
            if "labels" in data:
                self.image_labels = data["labels"]
                print(f"Labels loaded: {len(self.image_labels)} labels")
                print(f"Label distribution: {np.bincount(self.image_labels)}")
            else:
                # Try to infer labels from filename
                filename = os.path.basename(data_path)
                if "mnist" in filename.lower():
                    # For MNIST, we can't easily get labels without the original dataset
                    self.image_labels = None
                    print("No labels found in data file")
                elif "cifar" in filename.lower():
                    self.image_labels = None
                    print("No labels found in data file")

            print(
                f"Data loaded: {len(self.images)} images, {self.true_firing_rates.shape[1]} neurons"
            )
            print(f"Image shape: {self.images.shape}")
            print(f"Firing rates shape: {self.true_firing_rates.shape}")

        except Exception as e:
            print(f"Error loading data: {e}")
            return

    def predict_firing_rates(self):
        """Get encoder predictions"""
        print("Generating encoder predictions...")

        self.encoder.eval()
        predicted_rates = []

        with torch.no_grad():
            for i in range(0, len(self.images), 32):  # Process in batches
                batch_images = self.images[i : i + 32]
                # Fix: only add channel if needed
                if batch_images.ndim == 3:
                    batch_tensor = torch.tensor(
                        batch_images[:, None, :, :], dtype=torch.float32
                    ).to(self.device)
                elif batch_images.ndim == 4:
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

    def analyze_firing_rate_distributions(self):
        """Analyze firing rate distributions to detect issues"""
        print("\n=== FIRING RATE DISTRIBUTION ANALYSIS ===")

        # Basic statistics
        true_mean = np.mean(self.true_firing_rates, axis=0)
        true_std = np.std(self.true_firing_rates, axis=0)
        pred_mean = np.mean(self.predicted_firing_rates, axis=0)
        pred_std = np.std(self.predicted_firing_rates, axis=0)

        print(
            f"True firing rates - Mean: {np.mean(true_mean):.2f} ± {np.mean(true_std):.2f}"
        )
        print(
            f"Predicted firing rates - Mean: {np.mean(pred_mean):.2f} ± {np.mean(pred_std):.2f}"
        )

        # Check for constant predictions
        constant_neurons = []
        for i in range(self.predicted_firing_rates.shape[1]):
            if (
                np.std(self.predicted_firing_rates[:, i]) < 0.1
            ):  # Very low variance
                constant_neurons.append(i)

        print(
            f"Neurons with constant predictions (< 0.1 std): {len(constant_neurons)}/{self.predicted_firing_rates.shape[1]}"
        )
        if constant_neurons:
            print(
                f"Constant neuron indices: {constant_neurons[:10]}..."
            )  # Show first 10

        # Correlation analysis
        correlations = []
        for i in range(self.predicted_firing_rates.shape[1]):
            corr = np.corrcoef(
                self.true_firing_rates[:, i], self.predicted_firing_rates[:, i]
            )[0, 1]
            correlations.append(corr)

        correlations = np.array(correlations)
        print(
            f"Mean correlation between true and predicted: {np.mean(correlations):.3f}"
        )
        print(f"Correlation std: {np.std(correlations):.3f}")
        print(
            f"Neurons with correlation > 0.5: {np.sum(correlations > 0.5)}/{len(correlations)}"
        )

        # Plot distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # True vs Predicted scatter
        axes[0, 0].scatter(true_mean, pred_mean, alpha=0.6)
        axes[0, 0].plot(
            [0, max(true_mean)], [0, max(true_mean)], "r--", alpha=0.8
        )
        axes[0, 0].set_xlabel("True Mean Firing Rate")
        axes[0, 0].set_ylabel("Predicted Mean Firing Rate")
        axes[0, 0].set_title("Mean Firing Rates: True vs Predicted")
        axes[0, 0].grid(True, alpha=0.3)

        # Correlation histogram
        axes[0, 1].hist(correlations, bins=30, alpha=0.7, edgecolor="black")
        axes[0, 1].axvline(
            np.mean(correlations),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(correlations):.3f}",
        )
        axes[0, 1].set_xlabel("Correlation Coefficient")
        axes[0, 1].set_ylabel("Number of Neurons")
        axes[0, 1].set_title("Distribution of True-Predicted Correlations")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Firing rate distributions
        axes[1, 0].hist(
            true_mean, bins=30, alpha=0.7, label="True", edgecolor="black"
        )
        axes[1, 0].hist(
            pred_mean, bins=30, alpha=0.7, label="Predicted", edgecolor="black"
        )
        axes[1, 0].set_xlabel("Mean Firing Rate")
        axes[1, 0].set_ylabel("Number of Neurons")
        axes[1, 0].set_title("Distribution of Mean Firing Rates")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Standard deviation comparison
        axes[1, 1].scatter(true_std, pred_std, alpha=0.6)
        axes[1, 1].plot(
            [0, max(true_std)], [0, max(true_std)], "r--", alpha=0.8
        )
        axes[1, 1].set_xlabel("True Std Firing Rate")
        axes[1, 1].set_ylabel("Predicted Std Firing Rate")
        axes[1, 1].set_title("Std Firing Rates: True vs Predicted")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        self.save_plot("firing_rate_analysis.png")

        return {
            "true_mean": true_mean,
            "pred_mean": pred_mean,
            "correlations": correlations,
            "constant_neurons": constant_neurons,
        }

    def analyze_neuron_responsiveness(self):
        """Analyze how responsive neurons are to different images"""
        print("\n=== NEURON RESPONSIVENESS ANALYSIS ===")

        # Calculate responsiveness (how much firing rate varies across images)
        true_responsiveness = np.std(self.true_firing_rates, axis=0)
        pred_responsiveness = np.std(self.predicted_firing_rates, axis=0)

        # Find neurons with high true responsiveness but low predicted responsiveness
        high_true_low_pred = []
        for i in range(len(true_responsiveness)):
            if true_responsiveness[i] > np.percentile(
                true_responsiveness, 75
            ) and pred_responsiveness[i] < np.percentile(
                pred_responsiveness, 25
            ):
                high_true_low_pred.append(i)

        print(
            f"Neurons with high true responsiveness but low predicted responsiveness: {len(high_true_low_pred)}"
        )

        # Plot responsiveness comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        axes[0].scatter(true_responsiveness, pred_responsiveness, alpha=0.6)
        axes[0].plot(
            [0, max(true_responsiveness)],
            [0, max(true_responsiveness)],
            "r--",
            alpha=0.8,
        )
        axes[0].set_xlabel("True Responsiveness (Std)")
        axes[0].set_ylabel("Predicted Responsiveness (Std)")
        axes[0].set_title("Neuron Responsiveness: True vs Predicted")
        axes[0].grid(True, alpha=0.3)

        # Highlight problematic neurons
        if high_true_low_pred:
            axes[0].scatter(
                true_responsiveness[high_true_low_pred],
                pred_responsiveness[high_true_low_pred],
                color="red",
                s=50,
                alpha=0.8,
                label="High True, Low Pred",
            )
            axes[0].legend()

        # Responsiveness distribution
        axes[1].hist(
            true_responsiveness,
            bins=30,
            alpha=0.7,
            label="True",
            edgecolor="black",
        )
        axes[1].hist(
            pred_responsiveness,
            bins=30,
            alpha=0.7,
            label="Predicted",
            edgecolor="black",
        )
        axes[1].set_xlabel("Responsiveness (Std)")
        axes[1].set_ylabel("Number of Neurons")
        axes[1].set_title("Distribution of Neuron Responsiveness")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        self.save_plot("responsiveness_analysis.png")

        return {
            "true_responsiveness": true_responsiveness,
            "pred_responsiveness": pred_responsiveness,
            "high_true_low_pred": high_true_low_pred,
        }

    def test_image_classification_from_firing_rates(self):
        """Test if predicted firing rates contain enough information for image classification"""
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
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

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
            f"Best classifier for PREDICTED firing rates: {best_pred_classifier[0].replace('_pred', '')} ({best_pred_classifier[1]:.3f})"
        )
        print(
            f"Best classifier for TRUE firing rates: {best_true_classifier[0].replace('_true', '')} ({best_true_classifier[1]:.3f})"
        )

        if best_true_classifier[1] > 0:
            degradation = (
                (best_true_classifier[1] - best_pred_classifier[1])
                / best_true_classifier[1]
            ) * 100
            print(f"Performance degradation: {degradation:.1f}%")

        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Predicted vs True comparison
        classifier_names = [
            name.replace("_pred", "").replace("_true", "")
            for name in pred_accuracies.keys()
        ]
        pred_scores = list(pred_accuracies.values())
        true_scores = list(true_accuracies.values())

        x = np.arange(len(classifier_names))
        width = 0.35

        axes[0].bar(
            x - width / 2, pred_scores, width, label="Predicted", alpha=0.8
        )
        axes[0].bar(x + width / 2, true_scores, width, label="True", alpha=0.8)
        axes[0].set_xlabel("Classifier")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title(
            "Classification Accuracy: Predicted vs True Firing Rates"
        )
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(classifier_names, rotation=45, ha="right")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)

        # Add accuracy values on bars
        for i, (pred, true) in enumerate(zip(pred_scores, true_scores)):
            axes[0].text(
                i - width / 2,
                pred + 0.01,
                f"{pred:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            axes[0].text(
                i + width / 2,
                true + 0.01,
                f"{true:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Performance degradation
        degradations = [
            (true - pred) / true * 100 if true > 0 else 0
            for pred, true in zip(pred_scores, true_scores)
        ]
        axes[1].bar(classifier_names, degradations, alpha=0.8, color="orange")
        axes[1].set_xlabel("Classifier")
        axes[1].set_ylabel("Performance Degradation (%)")
        axes[1].set_title("Performance Degradation: (True - Predicted) / True")
        axes[1].set_xticklabels(classifier_names, rotation=45, ha="right")
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # Add degradation values on bars
        for i, deg in enumerate(degradations):
            axes[1].text(
                i, deg + 1, f"{deg:.1f}%", ha="center", va="bottom", fontsize=8
            )

        plt.tight_layout()
        self.save_plot("classifier_comparison.png")

        # Feature importance analysis for best classifier
        print(
            f"\nFeature importance analysis for best classifier ({best_pred_classifier[0].replace('_pred', '')}):"
        )

        # Retrain best classifier to get feature importance
        best_clf_name = best_pred_classifier[0].replace("_pred", "")
        best_clf = classifiers[best_clf_name]

        if hasattr(best_clf, "feature_importances_"):
            # For Random Forest
            if name.startswith("SVM") or name.startswith("MLP"):
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                best_clf.fit(X_train_scaled, y_train)
            else:
                best_clf.fit(X_train, y_train)

            feature_importance = best_clf.feature_importances_
            top_neurons = np.argsort(feature_importance)[
                -10:
            ]  # Top 10 neurons

            print("\nTop 10 most important neurons for classification:")
            for i, neuron_idx in enumerate(reversed(top_neurons)):
                print(
                    f"  {i + 1}. Neuron {neuron_idx}: importance = {feature_importance[neuron_idx]:.4f}"
                )

        return {
            "results": results,
            "best_pred_classifier": best_pred_classifier,
            "best_true_classifier": best_true_classifier,
        }

    def analyze_encoder_representations(self):
        """Analyze the learned representations using dimensionality reduction"""
        print("\n=== ENCODER REPRESENTATION ANALYSIS ===")

        # Check data variance first
        data_variance = np.var(self.predicted_firing_rates, axis=0)
        print("Data variance statistics:")
        print(f"  Mean variance across neurons: {np.mean(data_variance):.6f}")
        print(f"  Std variance across neurons: {np.std(data_variance):.6f}")
        print(f"  Min variance: {np.min(data_variance):.6f}")
        print(f"  Max variance: {np.max(data_variance):.6f}")

        # Use PCA with more components to get better analysis
        n_components = min(100, self.predicted_firing_rates.shape[1])
        pca = PCA(n_components=n_components)
        representations_pca = pca.fit_transform(self.predicted_firing_rates)

        # Analyze explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        print("PCA explained variance:")
        print(
            f"  First 5 components: {np.sum(explained_variance_ratio[:5]):.6f}"
        )
        print(
            f"  First 10 components: {np.sum(explained_variance_ratio[:10]):.6f}"
        )
        print(
            f"  First 20 components: {np.sum(explained_variance_ratio[:20]):.6f}"
        )

        # Find components needed for different variance thresholds
        for threshold in [0.5, 0.8, 0.9, 0.95]:
            n_comp = np.argmax(cumulative_variance >= threshold) + 1
            print(
                f"  Components needed for {threshold * 100}% variance: {n_comp}"
            )

        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(explained_variance_ratio) + 1),
            explained_variance_ratio,
            "b-",
            linewidth=1,
            alpha=0.7,
        )
        plt.xlabel("Component Number")
        plt.ylabel("Explained Variance Ratio")
        plt.title("PCA: Individual Component Variance")
        plt.yscale("log")
        plt.grid(True, alpha=0.3)
        self.save_plot("pca_analysis.png")

        # Additional analysis: check if the first few components are meaningful
        if len(explained_variance_ratio) > 0:
            print("\nFirst 10 components explained variance:")
            for i, var in enumerate(explained_variance_ratio[:10]):
                print(f"  Component {i + 1}: {var:.6f}")

        return {
            "pca_components": representations_pca,
            "explained_variance_ratio": explained_variance_ratio,
            "data_variance": data_variance,
        }

    def suggest_improvements(self, analysis_results):
        """Suggest improvements based on analysis results"""
        print("\n=== SUGGESTED IMPROVEMENTS ===")

        suggestions = []

        # Check for constant predictions
        if analysis_results.get("constant_neurons"):
            suggestions.append(
                {
                    "issue": "Many neurons have constant predictions",
                    "suggestions": [
                        "Increase model capacity (more layers/neurons)",
                        "Add regularization to prevent overfitting to average rates",
                        "Use different activation functions (e.g., ReLU instead of ELU)",
                        "Add batch normalization to prevent internal covariate shift",
                        "Consider using a different loss function (e.g., cosine similarity)",
                    ],
                }
            )

        # Check correlation
        correlations = analysis_results.get("correlations", [])
        if correlations and np.mean(correlations) < 0.3:
            suggestions.append(
                {
                    "issue": "Low correlation between true and predicted firing rates",
                    "suggestions": [
                        "Increase training epochs",
                        "Adjust learning rate schedule",
                        "Add data augmentation",
                        "Use curriculum learning (start with simpler patterns)",
                        "Consider ensemble methods",
                    ],
                }
            )

        # Check responsiveness
        responsiveness_results = analysis_results.get("responsiveness", {})
        if responsiveness_results.get("high_true_low_pred"):
            suggestions.append(
                {
                    "issue": "Neurons with high true responsiveness have low predicted responsiveness",
                    "suggestions": [
                        "Add skip connections to preserve fine-grained information",
                        "Use attention mechanisms",
                        "Implement progressive training (start with low-resolution, increase gradually)",
                        "Add auxiliary losses to encourage responsiveness",
                    ],
                }
            )

        # Check classification performance
        classification_results = analysis_results.get("classification", {})
        if classification_results:
            accuracy_pred = classification_results.get("accuracy_pred", 0)
            if accuracy_pred < 0.5:
                suggestions.append(
                    {
                        "issue": "Poor image classification from predicted firing rates",
                        "suggestions": [
                            "The encoder is not learning meaningful representations",
                            "Consider using contrastive learning",
                            "Add reconstruction loss as auxiliary task",
                            "Use pre-trained vision encoders and fine-tune",
                            "Implement multi-task learning with image classification",
                        ],
                    }
                )

        # Print suggestions
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n{i}. {suggestion['issue']}")
            for j, sub_suggestion in enumerate(suggestion["suggestions"], 1):
                print(f"   {j}. {sub_suggestion}")

        return suggestions

    def analyze_feature_scaling_and_separability(self):
        """Analyze feature scaling effects and investigate why linear SVM works well"""
        print("\n=== FEATURE SCALING AND LINEAR SEPARABILITY ANALYSIS ===")

        if self.image_labels is None:
            print("No image labels available. Skipping analysis.")
            return None

        import matplotlib.pyplot as plt
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        from sklearn.svm import SVC

        # Prepare data
        X_pred = self.predicted_firing_rates
        X_true = self.true_firing_rates
        y = self.image_labels

        X_train_pred, X_test_pred, y_train, y_test = train_test_split(
            X_pred, y, test_size=0.3, random_state=42, stratify=y
        )

        # Test different scaling methods
        scalers = {
            "No Scaling": None,
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
        }

        print("Testing different scaling methods with Linear SVM:")
        print("-" * 60)

        scaling_results = {}

        for scaler_name, scaler in scalers.items():
            if scaler is None:
                X_train_scaled = X_train_pred
                X_test_scaled = X_test_pred
            else:
                X_train_scaled = scaler.fit_transform(X_train_pred)
                X_test_scaled = scaler.transform(X_test_pred)

            # Train linear SVM
            svm = SVC(kernel="linear", random_state=42)
            svm.fit(X_train_scaled, y_train)
            y_pred = svm.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            scaling_results[scaler_name] = accuracy
            print(f"{scaler_name}: {accuracy:.3f}")

        # Analyze feature distributions
        print("\nFeature distribution analysis:")
        print("-" * 60)

        # Original feature statistics
        pred_mean = np.mean(X_pred, axis=0)
        pred_std = np.std(X_pred, axis=0)
        pred_min = np.min(X_pred, axis=0)
        pred_max = np.max(X_pred, axis=0)

        true_mean = np.mean(X_true, axis=0)
        true_std = np.std(X_true, axis=0)
        true_min = np.min(X_true, axis=0)
        true_max = np.max(X_true, axis=0)

        print("Predicted firing rates:")
        print(f"  Mean: {np.mean(pred_mean):.2f} ± {np.std(pred_mean):.2f}")
        print(f"  Std: {np.mean(pred_std):.2f} ± {np.std(pred_std):.2f}")
        print(f"  Range: [{np.mean(pred_min):.2f}, {np.mean(pred_max):.2f}]")
        print(
            f"  Coefficient of variation: {np.mean(pred_std / pred_mean):.3f}"
        )

        print("\nTrue firing rates:")
        print(f"  Mean: {np.mean(true_mean):.2f} ± {np.std(true_mean):.2f}")
        print(f"  Std: {np.mean(true_std):.2f} ± {np.std(true_std):.2f}")
        print(f"  Range: [{np.mean(true_min):.2f}, {np.mean(true_max):.2f}]")
        print(
            f"  Coefficient of variation: {np.mean(true_std / true_mean):.3f}"
        )

        # Analyze linear separability
        print("\nLinear separability analysis:")
        print("-" * 60)

        # Use PCA to visualize separability in 2D
        from sklearn.decomposition import PCA

        # Standardize for PCA
        scaler = StandardScaler()
        X_pred_scaled = scaler.fit_transform(X_pred)
        X_true_scaled = scaler.fit_transform(X_true)

        # PCA to 2D
        pca = PCA(n_components=2)
        X_pred_2d = pca.fit_transform(X_pred_scaled)
        X_true_2d = pca.fit_transform(X_true_scaled)

        # Calculate class separability metrics
        def calculate_separability(X, y):
            # Calculate Fisher's discriminant ratio
            classes = np.unique(y)
            if len(classes) < 2:
                return 0

            # Calculate between-class and within-class scatter
            overall_mean = np.mean(X, axis=0)
            between_class_scatter = 0
            within_class_scatter = 0

            for c in classes:
                class_mask = y == c
                class_data = X[class_mask]
                class_mean = np.mean(class_data, axis=0)
                class_size = np.sum(class_mask)

                # Between-class scatter
                diff = class_mean - overall_mean
                between_class_scatter += class_size * np.outer(diff, diff)

                # Within-class scatter
                for sample in class_data:
                    diff = sample - class_mean
                    within_class_scatter += np.outer(diff, diff)

            # Fisher's discriminant ratio
            if np.linalg.det(within_class_scatter) > 1e-10:
                fisher_ratio = np.trace(
                    np.linalg.inv(within_class_scatter) @ between_class_scatter
                )
                return fisher_ratio
            else:
                return 0

        separability_pred = calculate_separability(X_pred_scaled, y)
        separability_true = calculate_separability(X_true_scaled, y)

        print("Fisher's discriminant ratio (higher = better separability):")
        print(f"  Predicted firing rates: {separability_pred:.3f}")
        print(f"  True firing rates: {separability_true:.3f}")
        print(
            f"  Ratio (pred/true): {separability_pred / separability_true:.3f}"
        )

        # Analyze support vectors
        print("\nSupport vector analysis:")
        print("-" * 60)

        # Train linear SVM on scaled data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_pred)
        X_test_scaled = scaler.transform(X_test_pred)

        svm = SVC(kernel="linear", random_state=42)
        svm.fit(X_train_scaled, y_train)

        n_support_vectors = len(svm.support_vectors_)
        n_samples = len(X_train_scaled)
        support_ratio = n_support_vectors / n_samples

        print(f"Number of support vectors: {n_support_vectors}")
        print(f"Total training samples: {n_samples}")
        print(f"Support vector ratio: {support_ratio:.3f}")
        print(f"Margin size: {1 / np.linalg.norm(svm.coef_[0]):.6f}")

        # Feature importance from SVM weights
        feature_importance = np.abs(svm.coef_[0])
        top_features = np.argsort(feature_importance)[-10:]

        print("\nTop 10 most important features (neurons) for linear SVM:")
        for i, feat_idx in enumerate(reversed(top_features)):
            print(
                f"  {i + 1}. Neuron {feat_idx}: weight = {svm.coef_[0][feat_idx]:.4f}"
            )

        # Plotting
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Scaling comparison
        scaler_names = list(scaling_results.keys())
        scaler_accuracies = list(scaling_results.values())
        axes[0, 0].bar(scaler_names, scaler_accuracies, alpha=0.8)
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].set_title("Linear SVM Performance with Different Scaling")
        axes[0, 0].set_ylim(0, 1)
        for i, acc in enumerate(scaler_accuracies):
            axes[0, 0].text(
                i, acc + 0.01, f"{acc:.3f}", ha="center", va="bottom"
            )

        # 2. Feature distribution comparison
        axes[0, 1].hist(
            pred_std, bins=30, alpha=0.7, label="Predicted", edgecolor="black"
        )
        axes[0, 1].hist(
            true_std, bins=30, alpha=0.7, label="True", edgecolor="black"
        )
        axes[0, 1].set_xlabel("Feature Standard Deviation")
        axes[0, 1].set_ylabel("Number of Features")
        axes[0, 1].set_title("Feature Variance Distribution")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Feature importance distribution
        axes[0, 2].hist(
            feature_importance, bins=30, alpha=0.7, edgecolor="black"
        )
        axes[0, 2].set_xlabel("SVM Feature Weight (Absolute)")
        axes[0, 2].set_ylabel("Number of Features")
        axes[0, 2].set_title("SVM Feature Importance Distribution")
        axes[0, 2].grid(True, alpha=0.3)

        # 4. PCA visualization - Predicted
        axes[1, 0].scatter(
            X_pred_2d[:, 0], X_pred_2d[:, 1], c=y, cmap="tab10", alpha=0.6
        )
        axes[1, 0].set_xlabel("PC1")
        axes[1, 0].set_ylabel("PC2")
        axes[1, 0].set_title(
            f"Predicted Firing Rates (PCA)\nSeparability: {separability_pred:.3f}"
        )
        axes[1, 0].grid(True, alpha=0.3)

        # 5. PCA visualization - True
        axes[1, 1].scatter(
            X_true_2d[:, 0], X_true_2d[:, 1], c=y, cmap="tab10", alpha=0.6
        )
        axes[1, 1].set_xlabel("PC1")
        axes[1, 1].set_ylabel("PC2")
        axes[1, 1].set_title(
            f"True Firing Rates (PCA)\nSeparability: {separability_true:.3f}"
        )
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Feature correlation with labels
        feature_correlations = []
        for i in range(X_pred.shape[1]):
            corr = np.corrcoef(X_pred[:, i], y)[0, 1]
            feature_correlations.append(abs(corr))

        axes[1, 2].hist(
            feature_correlations, bins=30, alpha=0.7, edgecolor="black"
        )
        axes[1, 2].set_xlabel("|Correlation with Labels|")
        axes[1, 2].set_ylabel("Number of Features")
        axes[1, 2].set_title("Feature-Label Correlation Distribution")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        self.save_plot("feature_analysis.png")

        return {
            "scaling_results": scaling_results,
            "separability_pred": separability_pred,
            "separability_true": separability_true,
            "support_ratio": support_ratio,
            "feature_importance": feature_importance,
            "feature_correlations": feature_correlations,
        }

    def run_full_analysis(self, model_path, data_path=None):
        """Run complete encoder analysis"""
        print("=== ENCODER VERIFICATION AND ANALYSIS ===")

        # Load data
        self.load_encoder_and_data(model_path, data_path)

        # Check if data was loaded successfully
        if self.images is None:
            print("Failed to load data. Exiting.")
            return None

        # Generate predictions
        self.predict_firing_rates()

        # Run analyses
        results = {}

        # Firing rate analysis
        results["firing_rates"] = self.analyze_firing_rate_distributions()

        # Responsiveness analysis
        results["responsiveness"] = self.analyze_neuron_responsiveness()

        # Classification test
        classification_results = (
            self.test_image_classification_from_firing_rates()
        )
        if classification_results:
            results["classification"] = classification_results

        # Representation analysis (PCA only, no t-SNE)
        results["representations"] = self.analyze_encoder_representations()

        # Feature scaling and separability analysis
        results["feature_analysis"] = (
            self.analyze_feature_scaling_and_separability()
        )

        # Suggest improvements
        suggestions = self.suggest_improvements(results)
        results["suggestions"] = suggestions

        print("\n=== ANALYSIS COMPLETE ===")
        print(f"All plots saved to: {self.plots_dir}")

        return results


def main():
    """Main function to run encoder verification"""
    import re
    import argparse

    # Use argparse for proper argument handling
    parser = argparse.ArgumentParser(description="Verify encoder model performance")
    parser.add_argument("--model", type=str, help="Path to the encoder model (.pth file)")
    parser.add_argument("--data", type=str, help="Path to the data file (.npz file)")
    
    # Filter out Jupyter-specific arguments
    filtered_args = []
    for arg in sys.argv[1:]:
        if not arg.startswith("--f=") and not arg.startswith("-f"):
            filtered_args.append(arg)
    
    # If we have positional arguments (old style), handle them
    if filtered_args and not any(arg.startswith("--") for arg in filtered_args):
        # Old-style positional arguments
        model_path = filtered_args[0] if len(filtered_args) > 0 else None
        data_path = filtered_args[1] if len(filtered_args) > 1 else None
    else:
        # Use argparse
        try:
            args = parser.parse_args(filtered_args)
            model_path = args.model
            data_path = args.data
        except SystemExit:
            # argparse failed, fall back to automatic detection
            model_path = None
            data_path = None

    # If model path is provided and looks valid, use it
    if model_path and os.path.exists(model_path) and model_path.endswith('.pth'):
        print(f"Using specified encoder model: {model_path}")
        
        # Try to infer the dataset file from the model filename
        match = re.search(
            r"(synthdata_dataset-[^_]+_sta-[^_]+_n_neurons-\d+_n_images-\d+_datetime-\d+_\d+)",
            model_path,
        )
        if match and not data_path:
            dataset_stem = match.group(1)
            # Find the matching .npz file
            data_candidates = glob.glob(f"data/{dataset_stem}.npz")
            if data_candidates:
                data_path = data_candidates[0]
                print(f"Inferred data file: {data_path}")
            else:
                print(
                    f"Could not find data file for dataset stem: {dataset_stem}"
                )
                data_path = None
        elif not match and not data_path:
            print(
                "Could not parse dataset stem from model filename. Please provide data file as second argument if needed."
            )
    else:
        # Fall back to automatic detection: use latest model
        model_files = glob.glob("data/encoder_model_*.pth") + glob.glob(
            "data/resnet_encoder_model_*.pth"
        )
        if not model_files:
            print("No encoder models found in data/ directory")
            print("Available files in data/:")
            for f in glob.glob("data/*"):
                print(f"  {f}")
            return
        model_path = max(model_files, key=os.path.getctime)
        print(f"Using latest encoder model: {model_path}")

    # Create verifier and run analysis
    verifier = EncoderVerifier()
    results = verifier.run_full_analysis(model_path, data_path)

    return results


def verify_latest_encoder(data_dir="data"):
    """
    Convenience function to verify the latest encoder model.
    This bypasses command line argument parsing and is more reliable in Jupyter environments.
    """
    print("=== ENCODER VERIFICATION AND ANALYSIS (Auto-detection) ===")
    
    # Find latest encoder model
    model_files = glob.glob(f"{data_dir}/encoder_model_*.pth") + glob.glob(
        f"{data_dir}/resnet_encoder_model_*.pth"
    )
    
    if not model_files:
        print(f"No encoder models found in {data_dir}/ directory")
        print(f"Available files in {data_dir}/:")
        for f in glob.glob(f"{data_dir}/*"):
            print(f"  {f}")
        return None
    
    model_path = max(model_files, key=os.path.getctime)
    print(f"Found latest encoder model: {model_path}")
    
    # Create verifier and run analysis
    verifier = EncoderVerifier()
    results = verifier.run_full_analysis(model_path, data_path=None)
    
    return results


if __name__ == "__main__":
    main()
