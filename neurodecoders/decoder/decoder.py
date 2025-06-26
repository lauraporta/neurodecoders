import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import glob
import os
import re
import datetime
from pathlib import Path

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

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 8, 8)  # Reshape to 8x8x512
        x = self.deconv(x)
        # Use interpolation instead of adaptive pooling for MPS compatibility
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        return x

# ---- Lightning Module ----
class DecoderLightningModule(pl.LightningModule):
    def __init__(self, in_neurons: int, image_size: int = 64, learning_rate: float = 1e-3, weight_decay: float = 1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = SimpleDecoder(in_neurons, image_size)
        self.loss_fn = nn.MSELoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Store training history for plotting
        self.train_losses = []
        self.val_losses = []
        
        # Store predictions and targets for metrics calculation
        self.train_predictions = []
        self.train_targets = []
        self.val_predictions = []
        self.val_targets = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        
        # Store predictions and targets for metrics calculation
        self.train_predictions.append(pred.detach().cpu())
        self.train_targets.append(y.detach().cpu())
        
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        
        # Store predictions and targets for metrics calculation
        self.val_predictions.append(pred.detach().cpu())
        self.val_targets.append(y.detach().cpu())
        
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

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
    
    def calculate_metrics_from_stored_data(self, predictions_list, targets_list):
        """Calculate PSNR and correlation from stored predictions and targets"""
        if not predictions_list or not targets_list:
            return 0.0, 0.0
        
        # Concatenate all predictions and targets
        predictions = torch.cat(predictions_list, dim=0).numpy()
        targets = torch.cat(targets_list, dim=0).numpy()
        
        # Ensure proper shapes
        if predictions.ndim == 4:
            predictions = predictions.squeeze(1)  # Remove channel dimension
        if targets.ndim == 4:
            targets = targets.squeeze(1)  # Remove channel dimension
        
        # Calculate metrics per image
        n_images = predictions.shape[0]
        psnr_values = []
        correlation_values = []
        
        for i in range(n_images):
            pred_img = predictions[i]
            target_img = targets[i]
            
            # Calculate MSE for this image
            mse = np.mean((target_img - pred_img) ** 2)
            
            # Calculate PSNR for this image
            max_val = np.max(target_img)
            if mse > 0:
                psnr = 20 * np.log10(max_val / np.sqrt(mse))
            else:
                psnr = float('inf')  # Perfect reconstruction
            psnr_values.append(psnr)
            
            # Calculate correlation for this image
            correlation = np.corrcoef(target_img.flatten(), pred_img.flatten())[0, 1]
            correlation_values.append(correlation)
        
        # Average the metrics across all images
        mean_psnr = np.mean(psnr_values)
        mean_correlation = np.mean(correlation_values)
        
        return mean_psnr, mean_correlation
    
    def clear_stored_data(self):
        """Clear stored predictions and targets to free memory"""
        self.train_predictions.clear()
        self.train_targets.clear()
        self.val_predictions.clear()
        self.val_targets.clear()

# ---- Data Module ----
class DecoderDataModule(pl.LightningDataModule):
    def __init__(self, firing_rates, images, train_split=0.7, val_split=0.15, 
                 batch_size=32, num_workers=0):
        super().__init__()
        self.firing_rates = firing_rates
        self.images = images
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Create full dataset
        self.full_dataset = NeuralDecoderDataset(firing_rates, images)
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

# ---- Custom Callback for Streamlit ----
class StreamlitCallback(pl.Callback):
    def __init__(self, progress_callback=None, metrics_callback=None, plot_callback=None, 
                 psnr_callback=None, correlation_callback=None):
        super().__init__()
        self.progress_callback = progress_callback
        self.metrics_callback = metrics_callback
        self.plot_callback = plot_callback
        self.psnr_callback = psnr_callback
        self.correlation_callback = correlation_callback
        self.train_losses = []
        self.val_losses = []
        self.train_psnr = []
        self.val_psnr = []
        self.train_correlation = []
        self.val_correlation = []
        
    def on_train_epoch_end(self, trainer, pl_module):
        # Get current losses
        train_loss = trainer.callback_metrics.get('train_loss_epoch', 0)
        val_loss = trainer.callback_metrics.get('val_loss', 0)
        
        if isinstance(train_loss, torch.Tensor):
            train_loss = train_loss.item()
        if isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.item()
            
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        # Calculate PSNR and correlation from stored data
        if hasattr(pl_module, 'train_predictions') and hasattr(pl_module, 'val_predictions'):
            # Calculate train metrics from stored data
            train_psnr, train_corr = pl_module.calculate_metrics_from_stored_data(
                pl_module.train_predictions, pl_module.train_targets
            )
            self.train_psnr.append(train_psnr)
            self.train_correlation.append(train_corr)
            
            # Calculate validation metrics from stored data
            val_psnr, val_corr = pl_module.calculate_metrics_from_stored_data(
                pl_module.val_predictions, pl_module.val_targets
            )
            self.val_psnr.append(val_psnr)
            self.val_correlation.append(val_corr)
            
            # Clear stored data to free memory
            pl_module.clear_stored_data()
        
        # Update progress
        if self.progress_callback:
            progress = (trainer.current_epoch + 1) / trainer.max_epochs
            self.progress_callback(progress, f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}")
        
        # Update metrics
        if self.metrics_callback:
            best_val_loss = min(self.val_losses) if self.val_losses else val_loss
            self.metrics_callback(train_loss, val_loss, best_val_loss)
        
        # Update plots
        if self.plot_callback and len(self.train_losses) > 1:
            self.plot_callback(self.train_losses, self.val_losses)
        
        # Update PSNR plot
        if self.psnr_callback and len(self.train_psnr) > 1:
            self.psnr_callback(self.train_psnr, self.val_psnr)
        
        # Update correlation plot
        if self.correlation_callback and len(self.train_correlation) > 1:
            self.correlation_callback(self.train_correlation, self.val_correlation)

def load_latest_data(dataset_to_load):
    """Load the latest neural data file"""
    files = glob.glob(dataset_to_load)
    if not files:
        raise FileNotFoundError("No neural data files found in data/ directory")
    
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

def train_model_lightning(
    firing_rates, 
    images, 
    train_split=0.7, 
    val_split=0.15, 
    batch_size=32, 
    learning_rate=1e-3, 
    epochs=30, 
    early_stopping_patience=5,
    enable_progress_bar=True,
    log_every_n_steps=50,
    callbacks=None,
    progress_callback=None,
    metrics_callback=None,
    plot_callback=None,
    psnr_callback=None,
    correlation_callback=None
):
    """
    Train decoder using PyTorch Lightning
    
    Returns:
        trainer: The trained trainer object
        model: The trained model
        data_module: The data module
    """
    # Create data module
    data_module = DecoderDataModule(
        firing_rates=firing_rates,
        images=images,
        train_split=train_split,
        val_split=val_split,
        batch_size=batch_size
    )
    
    # Create model
    model = DecoderLightningModule(
        in_neurons=firing_rates.shape[1],
        image_size=images.shape[1],  # Assuming square images
        learning_rate=learning_rate
    )
    
    # Setup callbacks
    if callbacks is None:
        callbacks = []
    
    # Add default callbacks
    callbacks.extend([
        EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            mode='min',
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch'),
        StreamlitCallback(
            progress_callback=progress_callback,
            metrics_callback=metrics_callback,
            plot_callback=plot_callback,
            psnr_callback=psnr_callback,
            correlation_callback=correlation_callback
        )
    ])
    
    # Setup logger
    logger = TensorBoardLogger("data/lightning_logs", name="decoder")
    
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
        enable_checkpointing=False  # Disable checkpoints
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

def save_predictions(model, firing_rates, images, input_file_path, output_dir='data', dataset_to_load=None):
    """Save model predictions and reconstructed images"""
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
    
    # Generate predictions
    model.eval()
    with torch.no_grad():
        firing_rates_tensor = torch.tensor(firing_rates, dtype=torch.float32)
        predictions = model(firing_rates_tensor)
        predictions = predictions.squeeze().cpu().numpy()
    
    # Ensure images are in the correct format for visualization
    if images.ndim == 4:
        # Images are (N, C, H, W) - remove channel dimension
        images_vis = images.squeeze(1)  # Remove channel dimension
    elif images.ndim == 3:
        # Images are already (N, H, W)
        images_vis = images
    else:
        raise ValueError(f"Unexpected image shape: {images.shape}")
    
    # Ensure predictions are in the correct format
    if predictions.ndim == 4:
        # Predictions are (N, C, H, W) - remove channel dimension
        predictions_vis = predictions.squeeze(1)
    elif predictions.ndim == 3:
        # Predictions are already (N, H, W)
        predictions_vis = predictions
    else:
        raise ValueError(f"Unexpected prediction shape: {predictions.shape}")
    
    # Save predictions with same timestamp
    output_filename = f'decoder_predictions_{dataset_to_load.stem}.npz'
    output_path = os.path.join(output_dir, output_filename)
    
    np.savez(output_path,
             original_images=images_vis,
             reconstructed_images=predictions_vis,
             neural_responses=firing_rates,
             input_file=input_file_path,
             timestamp=timestamp)
    
    print(f"Predictions saved to: {output_path}")
    print(f"Reconstructed images shape: {predictions_vis.shape}")
    print(f"Mean reconstruction error: {np.mean((predictions_vis - images_vis) ** 2):.4f}")
    
    # Visualize some reconstructions
    n_samples = min(10, len(images_vis))
    fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))
    
    for i in range(n_samples):
        # Original image
        axes[0, i].imshow(images_vis[i], cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstructed image
        axes[1, i].imshow(predictions_vis[i], cmap='gray')
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'decoder_reconstructions_{timestamp}.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return output_path

def main(dataset_to_load):
    """Main function to run the decoder training"""
    print("=== Neural Decoder Training with PyTorch Lightning ===")
    
    # Load data
    print("Loading data...")
    images, firing_rates, data_file = load_latest_data(dataset_to_load)
    
    # Preprocess data
    images, firing_rates, H, W = preprocess_data(images, firing_rates)
    
    # Visualize data
    visualize_data(firing_rates, images)
    
    # Train with Lightning
    trainer, model, data_module = train_model_lightning(
        firing_rates=firing_rates,
        images=images,
        epochs=30,
        learning_rate=1e-3,
        early_stopping_patience=5
    )
    
    # Plot training results
    plot_training_results(model.train_losses, model.val_losses)
    
    # Save predictions
    save_predictions(model, firing_rates, images, data_file, dataset_to_load)
    
    # Save final model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'data/decoder_{dataset_to_load.stem}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    print("=== Lightning Training Complete ===")

if __name__ == "__main__":
    dataset_to_load = Path("data/synthdata_dataset-mnist_sta-perlin_noise_patterns,11,11_n_neurons-1000_n_images-1000_20250626_114534.npz")
    main(dataset_to_load)
