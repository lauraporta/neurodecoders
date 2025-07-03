import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import glob
import os
import sys
from pathlib import Path
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from neurodecoders.encoder.encoder import SimpleEncoder, load_latest_data


class EncoderVerifier:
    """
    Comprehensive encoder verification and analysis tool.
    Diagnoses encoder issues and evaluates its quality for image classification.
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.encoder = None
        self.images = None
        self.true_firing_rates = None
        self.predicted_firing_rates = None
        self.image_labels = None
        
    def load_encoder_and_data(self, model_path, data_path=None):
        """Load encoder model and corresponding data"""
        print(f"Loading encoder from: {model_path}")
        
        # Load encoder
        try:
            state_dict = torch.load(model_path, map_location=self.device)
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        
        # Determine output neurons
        if 'model.fc.6.weight' in state_dict:
            out_neurons = state_dict['model.fc.6.weight'].shape[0]
        elif 'fc.6.weight' in state_dict:
            out_neurons = state_dict['fc.6.weight'].shape[0]
        else:
            # Find the last fc layer
            fc_keys = [k for k in state_dict.keys() if 'fc' in k and 'weight' in k]
            if not fc_keys:
                print("No fc layers found in model")
                return
            last_fc_key = sorted(fc_keys)[-1]
            out_neurons = state_dict[last_fc_key].shape[0]
        
        self.encoder = SimpleEncoder(out_neurons)
        
        # Handle nested model structure
        if any(k.startswith('model.') for k in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
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
            model_name = os.path.basename(model_path).replace('.pth', '')
            print(f"Looking for data file matching model: {model_name}")
            
            # Try different patterns
            patterns = [
                f"data/synthdata_dataset-*{model_name.split('_datetime-')[0]}*.npz",
                "data/synthdata_dataset-*.npz"
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
            
            if 'images' in data:
                self.images = data['images']
            else:
                print("No 'images' key found in data file")
                return
                
            if 'responses' in data:
                self.true_firing_rates = data['responses']
            else:
                print("No 'responses' key found in data file")
                return
            
            # Extract image labels if available
            if 'labels' in data:
                self.image_labels = data['labels']
                print(f"Labels loaded: {len(self.image_labels)} labels")
                print(f"Label distribution: {np.bincount(self.image_labels)}")
            else:
                # Try to infer labels from filename
                filename = os.path.basename(data_path)
                if 'mnist' in filename.lower():
                    # For MNIST, we can't easily get labels without the original dataset
                    self.image_labels = None
                    print("No labels found in data file")
                elif 'cifar' in filename.lower():
                    self.image_labels = None
                    print("No labels found in data file")
            
            print(f"Data loaded: {len(self.images)} images, {self.true_firing_rates.shape[1]} neurons")
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
                batch_images = self.images[i:i+32]
                # Fix: only add channel if needed
                if batch_images.ndim == 3:
                    batch_tensor = torch.tensor(batch_images[:, None, :, :], dtype=torch.float32).to(self.device)
                elif batch_images.ndim == 4:
                    batch_tensor = torch.tensor(batch_images, dtype=torch.float32).to(self.device)
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
        
        print(f"True firing rates - Mean: {np.mean(true_mean):.2f} ± {np.mean(true_std):.2f}")
        print(f"Predicted firing rates - Mean: {np.mean(pred_mean):.2f} ± {np.mean(pred_std):.2f}")
        
        # Check for constant predictions
        constant_neurons = []
        for i in range(self.predicted_firing_rates.shape[1]):
            if np.std(self.predicted_firing_rates[:, i]) < 0.1:  # Very low variance
                constant_neurons.append(i)
        
        print(f"Neurons with constant predictions (< 0.1 std): {len(constant_neurons)}/{self.predicted_firing_rates.shape[1]}")
        if constant_neurons:
            print(f"Constant neuron indices: {constant_neurons[:10]}...")  # Show first 10
        
        # Correlation analysis
        correlations = []
        for i in range(self.predicted_firing_rates.shape[1]):
            corr = np.corrcoef(self.true_firing_rates[:, i], self.predicted_firing_rates[:, i])[0, 1]
            correlations.append(corr)
        
        correlations = np.array(correlations)
        print(f"Mean correlation between true and predicted: {np.mean(correlations):.3f}")
        print(f"Correlation std: {np.std(correlations):.3f}")
        print(f"Neurons with correlation > 0.5: {np.sum(correlations > 0.5)}/{len(correlations)}")
        
        # Plot distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # True vs Predicted scatter
        axes[0, 0].scatter(true_mean, pred_mean, alpha=0.6)
        axes[0, 0].plot([0, max(true_mean)], [0, max(true_mean)], 'r--', alpha=0.8)
        axes[0, 0].set_xlabel('True Mean Firing Rate')
        axes[0, 0].set_ylabel('Predicted Mean Firing Rate')
        axes[0, 0].set_title('Mean Firing Rates: True vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Correlation histogram
        axes[0, 1].hist(correlations, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(np.mean(correlations), color='red', linestyle='--', label=f'Mean: {np.mean(correlations):.3f}')
        axes[0, 1].set_xlabel('Correlation Coefficient')
        axes[0, 1].set_ylabel('Number of Neurons')
        axes[0, 1].set_title('Distribution of True-Predicted Correlations')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Firing rate distributions
        axes[1, 0].hist(true_mean, bins=30, alpha=0.7, label='True', edgecolor='black')
        axes[1, 0].hist(pred_mean, bins=30, alpha=0.7, label='Predicted', edgecolor='black')
        axes[1, 0].set_xlabel('Mean Firing Rate')
        axes[1, 0].set_ylabel('Number of Neurons')
        axes[1, 0].set_title('Distribution of Mean Firing Rates')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Standard deviation comparison
        axes[1, 1].scatter(true_std, pred_std, alpha=0.6)
        axes[1, 1].plot([0, max(true_std)], [0, max(true_std)], 'r--', alpha=0.8)
        axes[1, 1].set_xlabel('True Std Firing Rate')
        axes[1, 1].set_ylabel('Predicted Std Firing Rate')
        axes[1, 1].set_title('Std Firing Rates: True vs Predicted')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/encoder_firing_rate_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'true_mean': true_mean,
            'pred_mean': pred_mean,
            'correlations': correlations,
            'constant_neurons': constant_neurons
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
            if true_responsiveness[i] > np.percentile(true_responsiveness, 75) and \
               pred_responsiveness[i] < np.percentile(pred_responsiveness, 25):
                high_true_low_pred.append(i)
        
        print(f"Neurons with high true responsiveness but low predicted responsiveness: {len(high_true_low_pred)}")
        
        # Plot responsiveness comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        axes[0].scatter(true_responsiveness, pred_responsiveness, alpha=0.6)
        axes[0].plot([0, max(true_responsiveness)], [0, max(true_responsiveness)], 'r--', alpha=0.8)
        axes[0].set_xlabel('True Responsiveness (Std)')
        axes[0].set_ylabel('Predicted Responsiveness (Std)')
        axes[0].set_title('Neuron Responsiveness: True vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # Highlight problematic neurons
        if high_true_low_pred:
            axes[0].scatter(true_responsiveness[high_true_low_pred], 
                          pred_responsiveness[high_true_low_pred], 
                          color='red', s=50, alpha=0.8, label='High True, Low Pred')
            axes[0].legend()
        
        # Responsiveness distribution
        axes[1].hist(true_responsiveness, bins=30, alpha=0.7, label='True', edgecolor='black')
        axes[1].hist(pred_responsiveness, bins=30, alpha=0.7, label='Predicted', edgecolor='black')
        axes[1].set_xlabel('Responsiveness (Std)')
        axes[1].set_ylabel('Number of Neurons')
        axes[1].set_title('Distribution of Neuron Responsiveness')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/encoder_responsiveness_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'true_responsiveness': true_responsiveness,
            'pred_responsiveness': pred_responsiveness,
            'high_true_low_pred': high_true_low_pred
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
        
        # Train Random Forest classifier
        print("Training Random Forest classifier on predicted firing rates...")
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_classifier.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Classification accuracy using predicted firing rates: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Compare with true firing rates
        X_true = self.true_firing_rates
        X_true_train, X_true_test, y_train_true, y_test_true = train_test_split(
            X_true, y, test_size=0.3, random_state=42, stratify=y
        )
        
        rf_true = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_true.fit(X_true_train, y_train_true)
        y_pred_true = rf_true.predict(X_true_test)
        accuracy_true = accuracy_score(y_test_true, y_pred_true)
        
        print(f"\nClassification accuracy using TRUE firing rates: {accuracy_true:.3f}")
        print(f"Performance degradation: {((accuracy_true - accuracy) / accuracy_true * 100):.1f}%")
        
        # Feature importance analysis
        feature_importance = rf_classifier.feature_importances_
        top_neurons = np.argsort(feature_importance)[-10:]  # Top 10 neurons
        
        print(f"\nTop 10 most important neurons for classification:")
        for i, neuron_idx in enumerate(reversed(top_neurons)):
            print(f"  {i+1}. Neuron {neuron_idx}: importance = {feature_importance[neuron_idx]:.4f}")
        
        # Plot feature importance
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Feature importance histogram
        axes[0].hist(feature_importance, bins=30, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Feature Importance')
        axes[0].set_ylabel('Number of Neurons')
        axes[0].set_title('Distribution of Feature Importance')
        axes[0].grid(True, alpha=0.3)
        
        # Top neurons importance
        top_importance = feature_importance[top_neurons]
        axes[1].barh(range(len(top_neurons)), top_importance)
        axes[1].set_yticks(range(len(top_neurons)))
        axes[1].set_yticklabels([f'Neuron {idx}' for idx in top_neurons])
        axes[1].set_xlabel('Feature Importance')
        axes[1].set_title('Top 10 Most Important Neurons')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/encoder_classification_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'accuracy_pred': accuracy,
            'accuracy_true': accuracy_true,
            'feature_importance': feature_importance,
            'top_neurons': top_neurons
        }
    
    def analyze_encoder_representations(self):
        """Analyze the learned representations using dimensionality reduction"""
        print("\n=== ENCODER REPRESENTATION ANALYSIS ===")
        
        # Check data variance first
        data_variance = np.var(self.predicted_firing_rates, axis=0)
        print(f"Data variance statistics:")
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
        
        print(f"PCA explained variance:")
        print(f"  First 5 components: {np.sum(explained_variance_ratio[:5]):.6f}")
        print(f"  First 10 components: {np.sum(explained_variance_ratio[:10]):.6f}")
        print(f"  First 20 components: {np.sum(explained_variance_ratio[:20]):.6f}")
        
        # Find components needed for different variance thresholds
        for threshold in [0.5, 0.8, 0.9, 0.95]:
            n_comp = np.argmax(cumulative_variance >= threshold) + 1
            print(f"  Components needed for {threshold*100}% variance: {n_comp}")
        
        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'b-', linewidth=1, alpha=0.7)
        plt.xlabel('Component Number')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA: Individual Component Variance')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.savefig('data/encoder_pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional analysis: check if the first few components are meaningful
        if len(explained_variance_ratio) > 0:
            print(f"\nFirst 10 components explained variance:")
            for i, var in enumerate(explained_variance_ratio[:10]):
                print(f"  Component {i+1}: {var:.6f}")
        
        return {
            'pca_components': representations_pca,
            'explained_variance_ratio': explained_variance_ratio,
            'data_variance': data_variance
        }
    
    def suggest_improvements(self, analysis_results):
        """Suggest improvements based on analysis results"""
        print("\n=== SUGGESTED IMPROVEMENTS ===")
        
        suggestions = []
        
        # Check for constant predictions
        if analysis_results.get('constant_neurons'):
            suggestions.append({
                'issue': 'Many neurons have constant predictions',
                'suggestions': [
                    'Increase model capacity (more layers/neurons)',
                    'Add regularization to prevent overfitting to average rates',
                    'Use different activation functions (e.g., ReLU instead of ELU)',
                    'Add batch normalization to prevent internal covariate shift',
                    'Consider using a different loss function (e.g., cosine similarity)'
                ]
            })
        
        # Check correlation
        correlations = analysis_results.get('correlations', [])
        if correlations and np.mean(correlations) < 0.3:
            suggestions.append({
                'issue': 'Low correlation between true and predicted firing rates',
                'suggestions': [
                    'Increase training epochs',
                    'Adjust learning rate schedule',
                    'Add data augmentation',
                    'Use curriculum learning (start with simpler patterns)',
                    'Consider ensemble methods'
                ]
            })
        
        # Check responsiveness
        responsiveness_results = analysis_results.get('responsiveness', {})
        if responsiveness_results.get('high_true_low_pred'):
            suggestions.append({
                'issue': 'Neurons with high true responsiveness have low predicted responsiveness',
                'suggestions': [
                    'Add skip connections to preserve fine-grained information',
                    'Use attention mechanisms',
                    'Implement progressive training (start with low-resolution, increase gradually)',
                    'Add auxiliary losses to encourage responsiveness'
                ]
            })
        
        # Check classification performance
        classification_results = analysis_results.get('classification', {})
        if classification_results:
            accuracy_pred = classification_results.get('accuracy_pred', 0)
            accuracy_true = classification_results.get('accuracy_true', 0)
            if accuracy_pred < 0.5:
                suggestions.append({
                    'issue': 'Poor image classification from predicted firing rates',
                    'suggestions': [
                        'The encoder is not learning meaningful representations',
                        'Consider using contrastive learning',
                        'Add reconstruction loss as auxiliary task',
                        'Use pre-trained vision encoders and fine-tune',
                        'Implement multi-task learning with image classification'
                    ]
                })
        
        # Print suggestions
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n{i}. {suggestion['issue']}")
            for j, sub_suggestion in enumerate(suggestion['suggestions'], 1):
                print(f"   {j}. {sub_suggestion}")
        
        return suggestions
    
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
        results['firing_rates'] = self.analyze_firing_rate_distributions()
        
        # Responsiveness analysis
        results['responsiveness'] = self.analyze_neuron_responsiveness()
        
        # Classification test
        classification_results = self.test_image_classification_from_firing_rates()
        if classification_results:
            results['classification'] = classification_results
        
        # Representation analysis (PCA only, no t-SNE)
        results['representations'] = self.analyze_encoder_representations()
        
        # Suggest improvements
        suggestions = self.suggest_improvements(results)
        results['suggestions'] = suggestions
        
        print("\n=== ANALYSIS COMPLETE ===")
        print("All plots saved to data/ directory")
        
        return results


def main():
    """Main function to run encoder verification"""
    # Find the latest encoder model
    model_files = glob.glob("data/encoder_model_*.pth")
    if not model_files:
        print("No encoder models found in data/ directory")
        return
    
    latest_model = max(model_files, key=os.path.getctime)
    print(f"Using latest encoder model: {latest_model}")
    
    # Check for optional data_path argument
    data_path = None
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        print(f"Using data file: {data_path}")
    
    # Create verifier and run analysis
    verifier = EncoderVerifier()
    results = verifier.run_full_analysis(latest_model, data_path)
    
    return results


if __name__ == "__main__":
    main()
