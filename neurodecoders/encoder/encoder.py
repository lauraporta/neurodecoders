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
class NeuralDataset(Dataset):
    def __init__(self, images, firing_rates):
        self.images = torch.tensor(images[:, None, :, :], dtype=torch.float32)  # Add channel dim
        self.firing_rates = torch.tensor(firing_rates, dtype=torch.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.firing_rates[idx]

# ---- Model definition ----
class SimpleEncoder(nn.Module):
    def __init__(self, out_neurons):
        super().__init__()
        # Deeper convolutional layers with batch normalization
        self.conv = nn.Sequential(
            # Initial conv layer with larger kernel to reduce spatial dimensions
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Middle conv layers
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # Final pooling
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Fully connected layers with dropout
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, out_neurons)
        )

    def forward(self, x):
        x = self.conv(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return F.softplus(x)  # Non-negative firing rates

def load_latest_data():
    """Load the latest neural data file"""
    files = glob.glob('output/simulated_neural_data_1000neurons_1000images_*.npz')
    if not files:
        raise FileNotFoundError("No neural data files found in output/ directory")
    
    latest_file = max(files, key=os.path.getctime)
    data = np.load(latest_file)
    images = data['images']         # Expecting shape: (N, 1, H, W)
    firing_rates = data['responses']   # Shape: (N, C) - already in firing rate format
    
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

    return images, firing_rates

def visualize_data(firing_rates):
    """Visualize the firing rate data"""
    print("\nFiring rate statistics:")
    print(f"Mean firing rate: {firing_rates.mean():.3f}")
    print(f"Std firing rate: {firing_rates.std():.3f}")
    print(f"Min firing rate: {firing_rates.min():.3f}")
    print(f"Max firing rate: {firing_rates.max():.3f}")

    # Plot firing rate distribution
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(firing_rates.flatten(), bins=50)
    plt.title('Firing Rate Distribution')
    plt.xlabel('Firing Rate')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    plt.imshow(firing_rates[:100].T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Firing Rate')
    plt.title('Firing Rates for First 100 Images')
    plt.xlabel('Image Index')
    plt.ylabel('Neuron Index')
    plt.tight_layout()
    plt.show()

def create_data_loaders(images, firing_rates, train_split=0.7, val_split=0.15, batch_size=32):
    """Create train/val/test data loaders"""
    print("Splitting data into train/val/test...")
    full_dataset = NeuralDataset(images, firing_rates)
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
    """Train the encoder model"""
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
    """Evaluate the model on test set"""
    print("Evaluating on test set...")
    model.eval()
    total_loss = 0
    loss_fn = nn.MSELoss()
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * x.size(0)
    avg_test_loss = total_loss / len(test_loader.dataset)
    print(f"Test Loss = {avg_test_loss:.4f}")
    
    return avg_test_loss

def plot_training_results(train_losses, val_losses):
    """Plot training results"""
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def save_predictions(model, images, firing_rates, input_file_path, output_dir='data'):
    """Save predicted neural responses with the same timestamp as input file"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract timestamp from input filename
    # Expected format: simulated_neural_data_*neurons_*images_YYYYMMDD_HHMMSS.npz
    timestamp_match = re.search(r'(\d{8}_\d{6})\.npz$', input_file_path)
    if timestamp_match:
        timestamp = timestamp_match.group(1)
    else:
        # If no timestamp found, use current time
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Prepare images for prediction
    if images.ndim == 4:
        # Images are already [N, C, H, W]
        input_images = torch.tensor(images, dtype=torch.float32)
    else:
        # Images are [N, H, W], add channel dimension
        input_images = torch.tensor(images[:, None, :, :], dtype=torch.float32)
    
    # Run predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        input_images = input_images.to(device)
        predictions = model(input_images).cpu().numpy()
    
    # Save predictions with same timestamp
    output_filename = f'predicted_neural_responses_{timestamp}.npz'
    output_path = os.path.join(output_dir, output_filename)
    
    np.savez(output_path,
             predicted_responses=predictions,
             actual_responses=firing_rates,
             input_file=input_file_path,
             timestamp=timestamp)
    
    print(f"Predictions saved to: {output_path}")
    print(f"Predicted responses shape: {predictions.shape}")
    print(f"Mean predicted firing rate: {predictions.mean():.3f}")
    
    return output_path

def main():
    """Main function to run the encoder training"""
    # Load data
    print("Loading data...")
    images, firing_rates, data_file = load_latest_data()
    
    # Preprocess data
    images, firing_rates = preprocess_data(images, firing_rates)
    
    # Visualize data
    visualize_data(firing_rates)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(images, firing_rates)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleEncoder(out_neurons=firing_rates.shape[1]).to(device)
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, device)
    
    # Evaluate model
    test_loss = evaluate_model(model, test_loader, device)
    
    # Plot results
    plot_training_results(train_losses, val_losses)

if __name__ == "__main__":
    main()
