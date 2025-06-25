import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import glob
import os
import re
import datetime

# ---- Dataset class ----
class NeuralDecoderDataset(Dataset):
    def __init__(self, firing_rates, images):
        self.firing_rates = torch.tensor(firing_rates, dtype=torch.float32)
        self.images = torch.tensor(images[:, None, :, :], dtype=torch.float32)  # Add channel dim

    def __len__(self):
        return len(self.firing_rates)

    def __getitem__(self, idx):
        return self.firing_rates[idx], self.images[idx]

# ---- Model definition ----
class SimpleDecoder(nn.Module):
    def __init__(self, in_neurons, image_size=64):
        super().__init__()
        self.image_size = image_size
        
        # Fully connected layers to expand neural responses
        self.fc = nn.Sequential(
            nn.Linear(in_neurons, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 8 * 8 * 512),  # Reshape to 8x8x512
            nn.ReLU()
        )
        
        # Transposed convolutional layers with adaptive upsampling
        self.deconv = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Final layer to get single channel
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Output values between 0 and 1
        )
        
        # Adaptive pooling to handle different image sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((image_size, image_size))

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 8, 8)  # Reshape to 8x8x512
        x = self.deconv(x)
        # Use adaptive pooling to ensure correct output size
        x = self.adaptive_pool(x)
        return x

def load_latest_data():
    """Load the latest neural data file"""
    files = glob.glob('output/simulated_neural_data_*neurons_*images_*.npz')
    if not files:
        raise FileNotFoundError("No neural data files found in output/ directory")
    
    latest_file = max(files, key=os.path.getctime)
    data = np.load(latest_file)
    images = data['images']         # Expecting shape: (N, 1, H, W) or (N, H, W)
    firing_rates = data['responses']   # Shape: (N, C) - neural responses
    
    return images, firing_rates, latest_file

def preprocess_data(images, firing_rates):
    """Preprocess and validate the data"""
    print(f"Images shape: {images.shape}")
    print(f"Firing rates shape: {firing_rates.shape}")

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

    return images, firing_rates, H, W

def visualize_data(firing_rates, images):
    """Visualize the neural response data and sample images"""
    print("\nFiring rate statistics:")
    print(f"Mean firing rate: {firing_rates.mean():.3f}")
    print(f"Std firing rate: {firing_rates.std():.3f}")
    print(f"Min firing rate: {firing_rates.min():.3f}")
    print(f"Max firing rate: {firing_rates.max():.3f}")

    # Plot firing rate distribution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(firing_rates.flatten(), bins=50)
    plt.title('Firing Rate Distribution')
    plt.xlabel('Firing Rate')
    plt.ylabel('Count')

    plt.subplot(1, 3, 2)
    plt.imshow(firing_rates[:100].T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Firing Rate')
    plt.title('Firing Rates for First 100 Images')
    plt.xlabel('Image Index')
    plt.ylabel('Neuron Index')
    
    plt.subplot(1, 3, 3)
    plt.imshow(images[0], cmap='gray')
    plt.title('Sample Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def create_data_loaders(firing_rates, images, train_split=0.7, val_split=0.15, batch_size=32):
    """Create train/val/test data loaders"""
    print("Splitting data into train/val/test...")
    full_dataset = NeuralDecoderDataset(firing_rates, images)
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, device, epochs=30, learning_rate=1e-3, early_stopping_patience=5):
    """Train the decoder model"""
    print("Initializing model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    loss_fn = nn.MSELoss()

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                total_loss += loss.item() * x.size(0)
        avg_val_loss = total_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return train_losses, val_losses

def evaluate_model(model, test_loader, device):
    """Evaluate the trained model on test data"""
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            total_loss += loss.item() * x.size(0)
            
            predictions.append(pred.cpu().numpy())
            actuals.append(y.cpu().numpy())
    
    avg_test_loss = total_loss / len(test_loader.dataset)
    print(f"Test Loss: {avg_test_loss:.4f}")
    
    return avg_test_loss, np.concatenate(predictions), np.concatenate(actuals)

def plot_training_results(train_losses, val_losses):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def save_predictions(model, firing_rates, images, input_file_path, output_dir='data'):
    """Save model predictions and reconstructed images"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate predictions
    model.eval()
    with torch.no_grad():
        firing_rates_tensor = torch.tensor(firing_rates, dtype=torch.float32)
        predictions = model(firing_rates_tensor)
        predictions = predictions.squeeze().cpu().numpy()
    
    # Save predictions
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'decoder_predictions_{timestamp}.npz')
    
    np.savez(output_file,
             original_images=images,
             reconstructed_images=predictions,
             neural_responses=firing_rates,
             input_file=input_file_path,
             timestamp=timestamp)
    
    print(f"Predictions saved to: {output_file}")
    
    # Visualize some reconstructions
    n_samples = min(10, len(images))
    fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))
    
    for i in range(n_samples):
        # Original image
        axes[0, i].imshow(images[i], cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstructed image
        axes[1, i].imshow(predictions[i], cmap='gray')
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'decoder_reconstructions_{timestamp}.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return output_file

def main():
    """Main function to run the decoder training"""
    print("=== Neural Decoder Training ===")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    images, firing_rates, input_file = load_latest_data()
    print(f"Loaded data from: {input_file}")
    
    # Preprocess data
    images, firing_rates, H, W = preprocess_data(images, firing_rates)
    
    # Visualize data
    visualize_data(firing_rates, images)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(firing_rates, images)
    
    # Initialize model
    model = SimpleDecoder(in_neurons=firing_rates.shape[1], image_size=H).to(device)
    print(f"Model initialized with {firing_rates.shape[1]} input neurons and image size {H}")
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, device)
    
    # Plot training results
    plot_training_results(train_losses, val_losses)
    
    # Evaluate model
    test_loss, predictions, actuals = evaluate_model(model, test_loader, device)
    
    # Save predictions
    save_predictions(model, firing_rates, images, input_file)
    
    # Save trained model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'data/decoder_model_{timestamp}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    print("=== Training Complete ===")

if __name__ == "__main__":
    main()
