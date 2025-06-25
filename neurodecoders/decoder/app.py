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

# Add the decoder directory to the path so we can import from it
sys.path.append(os.path.dirname(__file__))

# Import the decoder functionality
from decoder import SimpleDecoder, NeuralDecoderDataset

# Configure Streamlit page
st.set_page_config(
    page_title="Neural Decoder Dashboard",
    page_icon="🔄",
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

def plot_reconstructions(original_images, reconstructed_images, n_samples=10):
    """Plot original vs reconstructed images"""
    n_samples = min(n_samples, len(original_images))
    fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))
    
    for i in range(n_samples):
        # Original image
        axes[0, i].imshow(original_images[i], cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstructed image
        axes[1, i].imshow(reconstructed_images[i], cmap='gray')
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    return fig

def train_decoder(images, firing_rates, train_split=0.7, val_split=0.15, 
                 batch_size=32, learning_rate=1e-3, epochs=30, early_stopping_patience=5,
                 progress_callback=None, metrics_callback=None):
    """Train the decoder model with real-time updates"""
    
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
    full_dataset = NeuralDecoderDataset(firing_rates, images)
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
    
    # Initialize model with correct image size
    model = SimpleDecoder(in_neurons=C, image_size=H).to(device)
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
            torch.save(model.state_dict(), 'data/best_decoder_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break
    
    # Evaluate on test set
    model.eval()
    total_test_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            total_test_loss += loss.item() * x.size(0)
            
            predictions.append(pred.cpu().numpy())
            actuals.append(y.cpu().numpy())
    
    avg_test_loss = total_test_loss / len(test_loader.dataset)
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    
    return model, train_losses, val_losses, avg_test_loss, predictions, actuals

def load_trained_model(images, firing_rates, model_path):
    """Load a trained model and generate reconstructions"""
    try:
        # Create a copy to avoid modifying the original
        images_copy = images.copy()
        
        # Handle 4D image input if present
        if images_copy.ndim == 4:
            N, _, H, W = images_copy.shape
            images_copy = images_copy[:, 0, :, :]  # Take first channel
        elif images_copy.ndim == 3:
            N, H, W = images_copy.shape
        else:
            raise ValueError(f"Unexpected image shape: {images_copy.shape}")
        
        # Validate shapes
        N_r, C = firing_rates.shape
        if N != N_r:
            raise ValueError(f"Mismatch: images have {N} samples but firing rates have {N_r}")
        
        # Initialize model with correct parameters
        model = SimpleDecoder(in_neurons=C, image_size=H).to(device)
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Generate reconstructions
        with torch.no_grad():
            test_firing_rates = torch.tensor(firing_rates, dtype=torch.float32).to(device)
            reconstructions = model(test_firing_rates).cpu().numpy()
            
            # Ensure proper shape - remove extra dimensions if present
            if reconstructions.ndim == 4:
                reconstructions = reconstructions.squeeze(1)  # Remove channel dimension if present
            elif reconstructions.ndim == 2:
                # If it's 2D, reshape to (N, H, W)
                reconstructions = reconstructions.reshape(N, H, W)
        
        return model, reconstructions
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def main():
    st.title("🧠 Neural Decoder Dashboard")
    st.markdown("Train a decoder to reconstruct images from neural responses")
    
    # Sidebar for configuration and training parameters
    st.sidebar.header("Configuration")
    
    # Load data
    images, firing_rates, input_file = load_data()
    
    if images is None or firing_rates is None:
        st.stop()
    
    # Display basic data info in sidebar
    st.sidebar.subheader("📊 Data Info")
    st.sidebar.write(f"**Images:** {len(images)}")
    st.sidebar.write(f"**Neurons:** {firing_rates.shape[1]}")
    st.sidebar.write(f"**Image Size:** {images.shape[-2]}x{images.shape[-1]}")
    
    # Training parameters in sidebar
    st.sidebar.subheader("🚀 Training Parameters")
    
    train_split = st.sidebar.slider("Training split", 0.5, 0.9, 0.7, 0.05)
    val_split = st.sidebar.slider("Validation split", 0.1, 0.3, 0.15, 0.05)
    batch_size = st.sidebar.selectbox("Batch size", [16, 32, 64, 128], index=1)
    learning_rate = st.sidebar.selectbox("Learning rate", [1e-4, 5e-4, 1e-3, 5e-3], index=2)
    epochs = st.sidebar.slider("Epochs", 10, 100, 30)
    early_stopping_patience = st.sidebar.slider("Early stopping patience", 3, 10, 5)
    
    # Main content area
    st.header("📁 Load Trained Model")
    
    # Check for available model files - only decoder models
    model_files = glob.glob('data/decoder_model_*.pth')
    model_files.extend(glob.glob('data/best_decoder_model.pth'))
    model_files.extend(glob.glob('data/decoder_*.pth'))
    model_files = sorted(list(set(model_files)))  # Remove duplicates
    
    if model_files:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_model = st.selectbox(
                "Select trained decoder model:",
                ["None"] + [os.path.basename(f) for f in model_files],
                index=0
            )
        
        with col2:
            load_button = st.button("🔄 Load Model & Visualize", type="primary")
        
        if selected_model != "None" and load_button:
            model_path = os.path.join('data', selected_model)
            with st.spinner("Loading model and generating reconstructions..."):
                model, reconstructions = load_trained_model(images, firing_rates, model_path)
                
                if model is not None and reconstructions is not None:
                    st.success(f"✅ Model loaded successfully: {selected_model}")
                    
                    # Display reconstructions
                    st.header("🖼️ Image Reconstructions")
                    
                    # Get some test images for visualization
                    if images.ndim == 4:
                        test_images = images[:10, 0] if images.shape[1] == 1 else images[:10, 0]
                    else:
                        test_images = images[:10]
                    
                    test_reconstructions = reconstructions[:10]
                    
                    fig = plot_reconstructions(test_images, test_reconstructions)
                    st.pyplot(fig)
                    
                    # Calculate reconstruction quality metrics
                    st.subheader("📊 Reconstruction Quality")
                    col1, col2, col3 = st.columns(3)
                    
                    # MSE between original and reconstructed
                    mse = np.mean((test_images - test_reconstructions) ** 2)
                    with col1:
                        st.metric("MSE", f"{mse:.4f}")
                    
                    # PSNR (Peak Signal-to-Noise Ratio)
                    max_val = np.max(test_images)
                    psnr = 20 * np.log10(max_val / np.sqrt(mse))
                    with col2:
                        st.metric("PSNR (dB)", f"{psnr:.2f}")
                    
                    # SSIM-like correlation
                    correlation = np.corrcoef(test_images.flatten(), test_reconstructions.flatten())[0, 1]
                    with col3:
                        st.metric("Correlation", f"{correlation:.3f}")
                    
                    # Show more reconstructions if requested
                    st.subheader("🔍 Detailed Reconstructions")
                    n_detailed = st.slider("Number of detailed reconstructions", 5, 20, 10)
                    
                    fig, axes = plt.subplots(2, n_detailed, figsize=(2*n_detailed, 4))
                    for i in range(n_detailed):
                        # Original image
                        axes[0, i].imshow(test_images[i], cmap='gray')
                        axes[0, i].set_title(f'Original {i+1}')
                        axes[0, i].axis('off')
                        
                        # Reconstructed image
                        axes[1, i].imshow(test_reconstructions[i], cmap='gray')
                        axes[1, i].set_title(f'Reconstructed {i+1}')
                        axes[1, i].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Save detailed reconstructions
                    if st.button("💾 Save Detailed Reconstructions"):
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        fig_path = f'data/detailed_reconstructions_{timestamp}.png'
                        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                        st.success(f"✅ Detailed reconstructions saved to: {fig_path}")
                else:
                    st.error("❌ Failed to load model or generate reconstructions")
    else:
        st.info("No trained decoder models found in data/ directory")
    
    # Training section
    st.header("🚀 Train Decoder")
    
    # Train button
    if st.button("🚀 Start Training", type="primary"):
        st.info("Training started... This may take a while.")
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create metrics display
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        train_loss_metric = metrics_col1.empty()
        val_loss_metric = metrics_col2.empty()
        best_val_metric = metrics_col3.empty()
        
        # Create placeholder for loss plot
        loss_plot_placeholder = st.empty()
        
        # Initialize loss tracking
        train_losses = []
        val_losses = []
        
        def update_progress(progress, status):
            progress_bar.progress(progress)
            status_text.text(status)
        
        def update_metrics(train_loss, val_loss, best_val_loss):
            train_loss_metric.metric("Train Loss", f"{train_loss:.4f}")
            val_loss_metric.metric("Val Loss", f"{val_loss:.4f}")
            best_val_metric.metric("Best Val Loss", f"{best_val_loss:.4f}")
            
            # Update loss plot in real-time
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if len(train_losses) > 1:  # Only plot if we have at least 2 points
                fig = plot_training_curves(train_losses, val_losses)
                loss_plot_placeholder.pyplot(fig)
                plt.close(fig)  # Close to prevent memory issues
        
        # Train the model
        try:
            model, train_losses, val_losses, test_loss, predictions, actuals = train_decoder(
                images, firing_rates, train_split, val_split, batch_size, 
                learning_rate, epochs, early_stopping_patience,
                progress_callback=update_progress,
                metrics_callback=update_metrics
            )
            
            st.success("✅ Training completed!")
            
            # Display results
            st.header("📊 Training Results")
            
            # Training curves (final version)
            st.subheader("📈 Training Curves")
            fig = plot_training_curves(train_losses, val_losses)
            st.pyplot(fig)
            
            # Test results
            st.subheader("🧪 Test Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Test Loss", f"{test_loss:.4f}")
                st.metric("Final Train Loss", f"{train_losses[-1]:.4f}")
                st.metric("Final Val Loss", f"{val_losses[-1]:.4f}")
            
            with col2:
                st.metric("Best Val Loss", f"{min(val_losses):.4f}")
                st.metric("Training Epochs", len(train_losses))
                st.metric("Model Parameters", sum(p.numel() for p in model.parameters()))
            
            # Reconstructions
            st.subheader("🖼️ Image Reconstructions")
            
            # Get some test images for visualization
            if images.ndim == 4:
                test_images = images[:10, 0] if images.shape[1] == 1 else images[:10, 0]
            else:
                test_images = images[:10]
            
            # Generate reconstructions for visualization
            model.eval()
            with torch.no_grad():
                test_firing_rates = torch.tensor(firing_rates[:10], dtype=torch.float32).to(device)
                test_reconstructions = model(test_firing_rates).cpu().numpy().squeeze()
            
            fig = plot_reconstructions(test_images, test_reconstructions)
            st.pyplot(fig)
            
            # Save results
            st.subheader("💾 Save Results")
            
            if st.button("💾 Save Model and Predictions"):
                # Save model
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = f'data/decoder_model_{timestamp}.pth'
                torch.save(model.state_dict(), model_path)
                
                # Save predictions
                predictions_file = f'data/decoder_predictions_{timestamp}.npz'
                np.savez(predictions_file,
                         original_images=images,
                         reconstructed_images=predictions,
                         neural_responses=firing_rates,
                         test_loss=test_loss,
                         input_file=input_file,
                         timestamp=timestamp)
                
                st.success(f"✅ Model saved to: {model_path}")
                st.success(f"✅ Predictions saved to: {predictions_file}")
                
                # Save reconstruction plot
                fig_path = f'data/decoder_reconstructions_{timestamp}.png'
                fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                st.success(f"✅ Reconstruction plot saved to: {fig_path}")
        
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
