import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ---- Load your .npz file ----
print("Loading data...")
data = np.load('output/simulated_neural_data_1000neurons_1000images_20250610_160958.npz')
images = data['images']         # Expecting shape: (N, 1, H, W)
firing_rates = data['responses']   # Shape: (N, C) - already in firing rate format

print(f"Images shape: {images.shape}")
print(f"Firing rates shape: {firing_rates.shape}")

# ---- Handle 4D image input if present ----
if images.ndim == 4:
    N, _, H, W = images.shape
    images = images[:, 0, :, :]  # Take first channel
elif images.ndim == 3:
    N, H, W = images.shape
else:
    raise ValueError(f"Unexpected image shape: {images.shape}")

# ---- Validate shapes ----
N_r, C = firing_rates.shape
if N != N_r:
    raise ValueError(f"Mismatch: images have {N} samples but firing rates have {N_r}")

# ---- Data validation and visualization ----
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

# ---- Create full dataset and split ----
print("Splitting data into train/val/test...")
full_dataset = NeuralDataset(images, firing_rates)
total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# ---- Training setup ----
print("Initializing model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleEncoder(out_neurons=C).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
loss_fn = nn.MSELoss()

# ---- Training loop ----
epochs = 30  # Increased number of epochs
train_losses = []
val_losses = []

print("Starting training...")
best_val_loss = float('inf')
patience = 5
patience_counter = 0

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
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# ---- Final evaluation on test set ----
print("Evaluating on test set...")
model.eval()
total_loss = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        total_loss += loss.item() * x.size(0)
avg_test_loss = total_loss / len(test_loader.dataset)
print(f"Test Loss = {avg_test_loss:.4f}")

# ---- Plot loss curves ----
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
