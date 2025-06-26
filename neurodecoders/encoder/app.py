import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import glob
import datetime
import os
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add the encoder directory to the path so we can import from it
sys.path.append(os.path.dirname(__file__))

# Import the encoder functionality
from encoder import SimpleEncoder, NeuralDataset, train_model_lightning

# Configure Streamlit page
st.set_page_config(
    page_title="Neural Encoder Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

def load_data():
    """Load neural data file selected by user from dropdown"""
    try:
        # Find all synthdata files
        files = glob.glob('data/synthdata_dataset-*.npz')
        if not files:
            st.error("No neural data files found in data/ directory. Please generate data first using the synthetic dashboard.")
            return None, None, None
        
        # Create a mapping of display names to file paths
        file_options = {}
        for file_path in files:
            # Extract meaningful info from filename for display
            filename = os.path.basename(file_path)
            # Remove the synthdata_dataset- prefix and .npz suffix
            display_name = filename.replace('synthdata_dataset-', '').replace('.npz', '')
            # Replace underscores with spaces for better readability
            display_name = display_name.replace('_', ' ')
            file_options[display_name] = file_path
        
        # Sort by creation time (newest first) for the dropdown
        sorted_files = sorted(file_options.items(), key=lambda x: os.path.getctime(x[1]), reverse=True)
        
        # Create dropdown
        selected_display_name = st.selectbox(
            "Select Dataset:",
            options=[name for name, _ in sorted_files],
            index=0,  # Default to newest file
            help="Choose a synthetic dataset to load"
        )
        
        # Get the selected file path
        selected_file = file_options[selected_display_name]
        
        data = np.load(selected_file)
        images = data['images']
        firing_rates = data['responses']
        
        return images, firing_rates, selected_file
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

def plot_training_curves(train_losses, val_losses):
    """Plot training and validation loss curves"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label='Train Loss', linewidth=2)
    ax.plot(val_losses, label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_firing_rate_distribution(firing_rates):
    """Plot firing rate distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    ax1.hist(firing_rates.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Firing Rate Distribution')
    ax1.set_xlabel('Firing Rate (Hz)')
    ax1.set_ylabel('Count')
    ax1.grid(True, alpha=0.3)
    
    # Heatmap
    im = ax2.imshow(firing_rates[:100].T, aspect='auto', cmap='viridis')
    ax2.set_title('Firing Rates for First 100 Images')
    ax2.set_xlabel('Image Index')
    ax2.set_ylabel('Neuron Index')
    plt.colorbar(im, ax=ax2, label='Firing Rate (Hz)')
    
    plt.tight_layout()
    return fig

def plot_predictions_vs_actual(pred, actual, n_samples=10):
    """Plot predicted vs actual firing rates"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    # Get the number of neurons (columns) in the data
    n_neurons = min(pred.shape[1], actual.shape[1])
    n_plots = min(n_samples, len(axes), n_neurons)
    
    for i in range(n_plots):
        # Plot all samples for this neuron
        axes[i].scatter(actual[:, i], pred[:, i], alpha=0.6, s=20)
        
        # Add diagonal line
        max_val = max(actual[:, i].max(), pred[:, i].max())
        axes[i].plot([0, max_val], [0, max_val], 'r--', alpha=0.8)
        
        axes[i].set_xlabel('Actual Firing Rate')
        axes[i].set_ylabel('Predicted Firing Rate')
        axes[i].set_title(f'Neuron {i+1}')
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def train_encoder_lightning(images, firing_rates, train_split=0.7, val_split=0.15, 
                           batch_size=32, learning_rate=1e-3, epochs=30, early_stopping_patience=5,
                           progress_callback=None, metrics_callback=None):
    """Train the encoder model using PyTorch Lightning with real-time updates"""
    
    # Handle 4D image input if present
    if images.ndim == 4:
        N, _, H, W = images.shape
        images = images[:, 0, :, :]  # Take first channel
    elif images.ndim == 3:
        N, H, W = images.shape
    else:
        raise ValueError(f"Unexpected image shape: {images.shape}")
    
    # Validate shapes
    N_r, C = firing_rates.shape
    if N != N_r:
        raise ValueError(f"Mismatch: images have {N} samples but firing rates have {N_r}")
    
    # Train with Lightning (using the existing callback structure)
    trainer, model, data_module = train_model_lightning(
        images=images,
        firing_rates=firing_rates,
        train_split=train_split,
        val_split=val_split,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        enable_progress_bar=False  # Disable Lightning's progress bar since we have Streamlit
    )
    
    # Update progress to 100% when training is complete
    if progress_callback:
        progress_callback(1.0, "Training completed!")
    
    # Get test predictions
    model.eval()
    test_predictions = []
    test_actuals = []
    
    with torch.no_grad():
        for batch in data_module.test_dataloader():
            x, y = batch
            pred = model(x)
            test_predictions.append(pred.cpu().numpy())
            test_actuals.append(y.cpu().numpy())
    
    test_predictions = np.concatenate(test_predictions, axis=0)
    test_actuals = np.concatenate(test_actuals, axis=0)
    
    # Calculate test loss
    test_loss = np.mean((test_predictions - test_actuals) ** 2)
    
    # Update final metrics
    if metrics_callback:
        final_train_loss = model.train_losses[-1] if model.train_losses else 0
        final_val_loss = model.val_losses[-1] if model.val_losses else 0
        metrics_callback(final_train_loss, final_val_loss, final_val_loss)
    
    return {
        'model': model,
        'train_losses': model.train_losses,
        'val_losses': model.val_losses,
        'test_loss': test_loss,
        'predictions': test_predictions,
        'actuals': test_actuals,
        'epochs_trained': len(model.train_losses)
    }

def main():
    st.title("🧠 Neural Encoder Dashboard")
    st.markdown("Train a neural encoder to map images to firing rates using PyTorch Lightning")
    
    # Sidebar controls
    st.sidebar.header("📊 Data & Model Settings")
    
    # Load data
    images, firing_rates, data_file = load_data()
    
    if images is None:
        st.stop()
    
    # Display data info
    st.sidebar.info(f"""
    **Data Info:**
    - Images: {images.shape}
    - Neurons: {firing_rates.shape[1]}
    - Mean firing rate: {firing_rates.mean():.2f} Hz
    - Max firing rate: {firing_rates.max():.2f} Hz
    """)
    
    # Training parameters
    st.sidebar.header("⚙️ Training Parameters")
    
    # Data split
    train_split = st.sidebar.slider("Training Split", 0.5, 0.9, 0.7, 0.05)
    val_split = st.sidebar.slider("Validation Split", 0.05, 0.3, 0.15, 0.05)
    test_split = 1.0 - train_split - val_split
    
    if test_split < 0:
        st.sidebar.error("Invalid split: train + val > 1.0")
        st.stop()
    
    st.sidebar.info(f"Test split: {test_split:.2f}")
    
    # Model parameters
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    learning_rate = st.sidebar.selectbox("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3], index=2)
    epochs = st.sidebar.slider("Max Epochs", 10, 100, 30, 5)
    early_stopping_patience = st.sidebar.slider("Early Stopping Patience", 3, 10, 5, 1)
    
    # Main content
    st.header("📈 Data Visualization")
    
    # Show firing rate distribution
    if st.button("Show Firing Rate Distribution"):
        fig = plot_firing_rate_distribution(firing_rates)
        st.pyplot(fig)
    
    st.header("🚀 Training")
    
    if st.button("Start Training", type="primary"):
        # Training progress
        st.subheader("Training Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create placeholders for metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        # Create placeholders for real-time metrics
        train_loss_placeholder = metric_col1.empty()
        val_loss_placeholder = metric_col2.empty()
        best_val_placeholder = metric_col3.empty()
        
        def update_progress(progress, status):
            progress_bar.progress(progress)
            status_text.text(status)
        
        def update_metrics(train_loss, val_loss, best_val_loss):
            train_loss_placeholder.metric("Train Loss", f"{train_loss:.4f}")
            val_loss_placeholder.metric("Val Loss", f"{val_loss:.4f}")
            best_val_placeholder.metric("Best Val Loss", f"{best_val_loss:.4f}")
        
        try:
            # Train the model with Lightning
            with st.spinner("Training encoder with PyTorch Lightning..."):
                results = train_encoder_lightning(
                    images=images,
                    firing_rates=firing_rates,
                    train_split=train_split,
                    val_split=val_split,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    early_stopping_patience=early_stopping_patience,
                    progress_callback=update_progress,
                    metrics_callback=update_metrics
                )
            
            progress_bar.progress(1.0)
            status_text.text("Training completed!")
            
            # Display results
            st.subheader("📊 Final Evaluation")
            st.success(f"Test Loss: {results['test_loss']:.4f}")
            
            # Plot final training curves
            st.subheader("📈 Final Training Curves")
            fig = plot_training_curves(results['train_losses'], results['val_losses'])
            st.pyplot(fig)
            
            # Plot predictions vs actual
            st.subheader("🎯 Predictions vs Actual")
            fig = plot_predictions_vs_actual(results['predictions'], results['actuals'])
            st.pyplot(fig)
            
            # Save training info to data folder
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            model_info = {
                'test_loss': results['test_loss'],
                'train_losses': results['train_losses'],
                'val_losses': results['val_losses'],
                'epochs_trained': results['epochs_trained'],
                'data_file': data_file,
                'timestamp': timestamp
            }
            
            training_info_path = f'data/encoder_training_info_{timestamp}.npy'
            np.save(training_info_path, model_info)
            st.info(f"Training info saved to: {training_info_path}")
            
            # Save model
            model_path = f'data/encoder_model_{os.path.splitext(os.path.basename(data_file))[0]}.pth'
            torch.save(results['model'].state_dict(), model_path)
            st.info(f"Model saved to: {model_path}")
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main() 