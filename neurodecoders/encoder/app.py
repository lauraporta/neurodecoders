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

# Add the encoder directory to the path so we can import from it
sys.path.append(os.path.dirname(__file__))

# Import the encoder functionality
from encoder import SimpleEncoder, NeuralDataset

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
    """Load the latest neural data file"""
    try:
        files = glob.glob('output/simulated_neural_data_*neurons_*images_*.npz')
        if not files:
            st.error("No neural data files found in output/ directory. Please generate data first using the synthetic dashboard.")
            return None, None, None
        
        latest_file = max(files, key=os.path.getctime)
        st.success(f"Loaded data from: {latest_file}")
        
        data = np.load(latest_file)
        images = data['images']
        firing_rates = data['responses']
        
        return images, firing_rates, latest_file
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
    
    for i in range(min(n_samples, len(axes))):
        axes[i].scatter(actual[i], pred[i], alpha=0.6, s=20)
        axes[i].plot([0, max(actual[i].max(), pred[i].max())], 
                    [0, max(actual[i].max(), pred[i].max())], 'r--', alpha=0.8)
        axes[i].set_xlabel('Actual Firing Rate')
        axes[i].set_ylabel('Predicted Firing Rate')
        axes[i].set_title(f'Sample {i+1}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def train_encoder(images, firing_rates, train_split=0.7, val_split=0.15, 
                 batch_size=32, learning_rate=1e-3, epochs=30, early_stopping_patience=5,
                 progress_callback=None, metrics_callback=None):
    """Train the encoder model with real-time updates"""
    
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
    
    # Create dataset and split
    full_dataset = NeuralDataset(images, firing_rates)
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = SimpleEncoder(out_neurons=C).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    loss_fn = nn.MSELoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * x.size(0)
        
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                total_val_loss += loss.item() * x.size(0)
        
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        
        # Update progress and metrics in real-time
        if progress_callback:
            progress = (epoch + 1) / epochs
            progress_callback(progress, f"Epoch {epoch+1}/{epochs}")
        
        if metrics_callback:
            metrics_callback(avg_train_loss, avg_val_loss, best_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model to data folder
            torch.save(model.state_dict(), 'data/best_encoder_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if progress_callback:
                    progress_callback(1.0, f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('data/best_encoder_model.pth'))
    
    # Test set evaluation
    model.eval()
    total_test_loss = 0
    all_predictions = []
    all_actuals = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            total_test_loss += loss.item() * x.size(0)
            all_predictions.append(pred.cpu())
            all_actuals.append(y.cpu())
    
    avg_test_loss = total_test_loss / len(test_loader.dataset)
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': avg_test_loss,
        'predictions': torch.cat(all_predictions, dim=0).numpy(),
        'actuals': torch.cat(all_actuals, dim=0).numpy(),
        'epochs_trained': len(train_losses)
    }

def main():
    st.title("🧠 Neural Encoder Dashboard")
    st.markdown("Train a neural encoder to map images to firing rates")
    
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
        
        # Create a placeholder for real-time loss plot
        loss_plot_placeholder = st.empty()
        
        # Initialize loss history for plotting
        train_losses = []
        val_losses = []
        
        def update_progress(progress, status):
            progress_bar.progress(progress)
            status_text.text(status)
        
        def update_metrics(train_loss, val_loss, best_val_loss):
            nonlocal train_losses, val_losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Update metric displays
            train_loss_placeholder.metric("Train Loss", f"{train_loss:.4f}")
            val_loss_placeholder.metric("Val Loss", f"{val_loss:.4f}")
            best_val_placeholder.metric("Best Val Loss", f"{best_val_loss:.4f}")
            
            # Update real-time loss plot
            if len(train_losses) > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(train_losses, label='Train Loss', linewidth=2, color='blue')
                ax.plot(val_losses, label='Validation Loss', linewidth=2, color='red')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('MSE Loss')
                ax.set_title('Training Progress (Real-time)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                loss_plot_placeholder.pyplot(fig)
                plt.close(fig)
        
        try:
            # Train the model with real-time updates
            with st.spinner("Training encoder..."):
                results = train_encoder(
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
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            st.exception(e)
    
    # Model loading and inference
    st.header("🔍 Model Inference")
    
    # Check for saved models in data folder
    model_files = glob.glob('data/best_encoder_model.pth')
    if model_files:
        st.success("Found trained model!")
        
        if st.button("Load Model and Run Inference"):
            try:
                # Load model from data folder
                model = SimpleEncoder(out_neurons=firing_rates.shape[1]).to(device)
                model.load_state_dict(torch.load('data/best_encoder_model.pth'))
                model.eval()
                
                # Run inference on a few samples
                with torch.no_grad():
                    # Handle image dimensions properly
                    if images.ndim == 4:
                        # Images are already [N, C, H, W], just take first 5
                        sample_images = torch.tensor(images[:5], dtype=torch.float32).to(device)
                    else:
                        # Images are [N, H, W], add channel dimension
                        sample_images = torch.tensor(images[:5, None, :, :], dtype=torch.float32).to(device)
                    
                    predictions = model(sample_images).cpu().numpy()
                    actuals = firing_rates[:5]
                
                # Save predictions for all data
                from encoder import save_predictions
                predictions_file = save_predictions(model, images, firing_rates, data_file)
                st.success(f"Predictions saved to: {predictions_file}")
                
                # Display results
                st.subheader("Sample Predictions")
                fig, axes = plt.subplots(2, 5, figsize=(20, 8))
                
                for i in range(5):
                    # Show image - handle different dimensions
                    if images.ndim == 4:
                        img_display = images[i, 0]  # Take first channel if 4D
                    else:
                        img_display = images[i]
                    
                    axes[0, i].imshow(img_display, cmap='gray')
                    axes[0, i].set_title(f'Sample {i+1}')
                    axes[0, i].axis('off')
                    
                    # Show predictions vs actual
                    axes[1, i].scatter(actuals[i], predictions[i], alpha=0.6)
                    axes[1, i].plot([0, max(actuals[i].max(), predictions[i].max())], 
                                  [0, max(actuals[i].max(), predictions[i].max())], 'r--')
                    axes[1, i].set_xlabel('Actual')
                    axes[1, i].set_ylabel('Predicted')
                    axes[1, i].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Inference failed: {str(e)}")
                st.exception(e)
    else:
        st.info("No trained model found. Train a model first to run inference.")

if __name__ == "__main__":
    main() 