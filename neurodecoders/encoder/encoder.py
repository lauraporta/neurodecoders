import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import glob
import os
import re
import datetime
from pathlib import Path


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
            nn.Linear(2048, out_neurons),
            nn.ELU(),
        )

    def forward(self, x):
        x = self.conv(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x + 1

# ---- Lightning Module ----
class EncoderLightningModule(pl.LightningModule):
    def __init__(self, out_neurons: int, learning_rate: float = 1e-3, weight_decay: float = 1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = SimpleEncoder(out_neurons)
        self.loss_fn = nn.MSELoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Store training history for plotting
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        
        # Log validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        
        # Log test loss
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        return optimizer

    def on_train_epoch_end(self):
        # Store losses for plotting
        train_loss = self.trainer.callback_metrics.get('train_loss_epoch', 0)
        val_loss = self.trainer.callback_metrics.get('val_loss', 0)
        
        if isinstance(train_loss, torch.Tensor):
            train_loss = train_loss.item()
        if isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.item()
            
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

# ---- Data Module ----
class NeuralDataModule(pl.LightningDataModule):
    def __init__(self, images, firing_rates, train_split=0.7, val_split=0.15, 
                 batch_size=32, num_workers=0):
        super().__init__()
        self.images = images
        self.firing_rates = firing_rates
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Create full dataset
        self.full_dataset = NeuralDataset(images, firing_rates)
        self.setup_splits()

    def setup_splits(self):
        """Setup train/val/test splits"""
        total_size = len(self.full_dataset)
        train_size = int(self.train_split * total_size)
        val_size = int(self.val_split * total_size)
        test_size = total_size - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.full_dataset, 
            [train_size, val_size, test_size], 
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

def load_latest_data(dataset_to_load):
    """Load the latest neural data file"""
    files = glob.glob(dataset_to_load)
    if not files:
        raise FileNotFoundError("No neural data files found in data/ directory")
    
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

def train_model_lightning(
    images, 
    firing_rates, 
    train_split=0.7, 
    val_split=0.15, 
    batch_size=32, 
    learning_rate=1e-3, 
    epochs=30, 
    enable_progress_bar=True,
    log_every_n_steps=50,
    callbacks=None
):
    """
    Train encoder using PyTorch Lightning
    
    Returns:
        trainer: The trained trainer object
        model: The trained model
        data_module: The data module
    """
    # Create data module
    data_module = NeuralDataModule(
        images=images,
        firing_rates=firing_rates,
        train_split=train_split,
        val_split=val_split,
        batch_size=batch_size
    )
    
    # Create model
    model = EncoderLightningModule(
        out_neurons=firing_rates.shape[1],
        learning_rate=learning_rate
    )
    
    # Setup callbacks
    if callbacks is None:
        callbacks = []
    
    # Add default callbacks (without early stopping)
    callbacks.extend([
        LearningRateMonitor(logging_interval='epoch')
    ])
    
    # Setup logger
    logger = TensorBoardLogger("data/lightning_logs", name="encoder")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=enable_progress_bar,
        log_every_n_steps=log_every_n_steps,
        accelerator='cpu' if torch.backends.mps.is_available() else 'auto',  # Force CPU on MPS to avoid compatibility issues
        devices=1 if torch.backends.mps.is_available() else 'auto',
        deterministic=False,
        enable_checkpointing=True
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    # Test the model
    trainer.test(model, data_module)
    
    return trainer, model, data_module

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

def save_predictions(model, images, firing_rates, input_file_path, output_dir='data', dataset_to_load=None):
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
    output_filename = f'encoder_predictions_{dataset_to_load.stem}.npz'
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

def main(dataset_to_load):
    """Main function to run the encoder training"""
    print("=== Neural Encoder Training with PyTorch Lightning ===")
    
    # Load data
    print("Loading data...")
    images, firing_rates, data_file = load_latest_data(dataset_to_load)
    
    # Preprocess data
    images, firing_rates = preprocess_data(images, firing_rates)
    
    # Visualize data
    visualize_data(firing_rates)
    
    # Train with Lightning
    trainer, model, data_module = train_model_lightning(
        images=images,
        firing_rates=firing_rates,
        epochs=30,
        learning_rate=1e-3,
        enable_progress_bar=True
    )
    
    # Plot training results
    plot_training_results(model.train_losses, model.val_losses)
    
    # Save predictions
    save_predictions(model, images, firing_rates, data_file, dataset_to_load)
    
    # Save final model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'data/lightning_encoder_model_{timestamp}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    print("=== Lightning Training Complete ===")

if __name__ == "__main__":
    dataset_to_load = Path("data/synthdata_dataset-mnist_sta-perlin_noise_patterns,11,11_n_neurons-1000_n_images-1000_20250626_114534.npz")
    main(dataset_to_load)
