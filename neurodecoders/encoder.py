import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---- Load your .npz file ----
print("Loading data...")
data = np.load('output/simulated_neural_data_200neurons_1trials_1000images.npz')
images = data['images']         # Expecting shape: (N, 1, H, W)
responses = data['responses']   # Shape: (1, N, C, T)

print(f"Images shape: {images.shape}")
print(f"Responses shape: {responses.shape}")

# ---- Handle 4D image input if present ----
if images.ndim == 4:
    N, _, H, W = images.shape
    images = images[:, 0, :, :]  # Take first channel
elif images.ndim == 3:
    N, H, W = images.shape
else:
    raise ValueError(f"Unexpected image shape: {images.shape}")

# ---- Adjust responses shape ----
if responses.shape[0] == 1:
    responses = responses[0]  # Remove singleton trial dimension

N_r, C, T = responses.shape
if N != N_r:
    raise ValueError(f"Mismatch: images have {N} samples but responses have {N_r}")

# ---- Convert responses to firing rates ----
print("Converting spike trains to firing rates...")
firing_rates = responses.mean(axis=-1)  # Shape: (N, C)

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
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(32, out_neurons)

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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# ---- Training loop ----
epochs = 10
train_losses = []
val_losses = []

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
