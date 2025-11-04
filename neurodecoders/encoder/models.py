"""
Neural encoder model architectures.

This module contains different neural network architectures for encoding
images to neural firing rates.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class Simple3LayerEncoder(nn.Module):
    """
    Simple 3-layer convolutional encoder with spatial readout matching the paper.
    
    Architecture:
    - 3 convolutional layers to extract features at multiple scales
    - Spatial readout layer that learns RF position for each neuron
    - Linear readout from features at each neuron's RF position
    
    This matches the paper's approach: "three layer neural network that extracts 
    intermediate image features and a readout layer that learns the position of 
    each cell's receptive field in the monitor, extracts the intermediate features 
    at that point and linearly predicts a cell response"
    """
    
    def __init__(self, out_neurons, image_height=32, image_width=32, learn_positions=True):
        super().__init__()
        self.out_neurons = out_neurons
        self.image_height = image_height
        self.image_width = image_width
        self.learn_positions = learn_positions
        
        # Three convolutional layers as per paper
        # Keep spatial dimensions to enable spatial readout
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),  # 32x32 -> 32x32
            nn.ReLU(),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),  # 32x32 -> 32x32
            nn.ReLU(),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),  # 32x32 -> 32x32
            nn.ReLU(),
        )
        
        # Spatial readout: learnable RF positions for each neuron
        if learn_positions:
            # Initialize RF positions randomly within image bounds
            init_x = torch.rand(out_neurons) * (image_width - 1)
            init_y = torch.rand(out_neurons) * (image_height - 1)
            
            # Learnable parameters for RF positions
            self.rf_x = nn.Parameter(init_x)
            self.rf_y = nn.Parameter(init_y)
        else:
            # Fixed random positions (for baseline comparison)
            self.register_buffer('rf_x', torch.rand(out_neurons) * (image_width - 1))
            self.register_buffer('rf_y', torch.rand(out_neurons) * (image_height - 1))
        
        # Linear readout from features at each RF position
        # Each neuron gets its own linear weights across all feature channels
        self.feature_readout = nn.Linear(64, out_neurons)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Extract features through 3 conv layers
        features1 = self.conv1(x)  # (B, 16, H, W)
        features2 = self.conv2(features1)  # (B, 32, H, W)
        features3 = self.conv3(features2)  # (B, 64, H, W)
        
        # Spatial readout: extract features at each neuron's RF position
        # Use bilinear interpolation to sample from continuous positions
        
        # Normalize RF positions to [-1, 1] for grid_sample
        norm_x = 2.0 * self.rf_x / (self.image_width - 1) - 1.0
        norm_y = 2.0 * self.rf_y / (self.image_height - 1) - 1.0
        
        # Create sampling grid: (B, N_neurons, 1, 2) where last dim is (x, y)
        grid = torch.stack([norm_x, norm_y], dim=-1)  # (N_neurons, 2)
        grid = grid.unsqueeze(0).unsqueeze(2)  # (1, N_neurons, 1, 2)
        grid = grid.expand(batch_size, -1, -1, -1)  # (B, N_neurons, 1, 2)
        
        # Sample features at RF positions using bilinear interpolation
        # grid_sample expects (B, C, H, W) input and (B, H_out, W_out, 2) grid
        sampled_features = torch.nn.functional.grid_sample(
            features3, grid, mode='bilinear', padding_mode='border', align_corners=True
        )  # (B, 64, N_neurons, 1)
        
        # Reshape to (B, N_neurons, 64)
        sampled_features = sampled_features.squeeze(-1).permute(0, 2, 1)
        
        # Linear readout for each neuron
        outputs = self.feature_readout(sampled_features)  # (B, N_neurons, N_neurons)
        
        # Take diagonal elements (each neuron's own output)
        # This implements independent linear readouts per neuron
        firing_rates = torch.diagonal(outputs, dim1=1, dim2=2)  # (B, N_neurons)
        
        return firing_rates


class ResNetMixin:
    """
    Mixin class providing common functionality for ResNet-based encoders.
    """

    def unfreeze_backbone(self, num_layers=None):
        """
        Unfreeze the last num_layers of the backbone for fine-tuning.
        If num_layers is None, unfreezes all layers.
        """
        if num_layers is None:
            # Unfreeze all backbone parameters
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("Unfroze all backbone layers")
        else:
            # Unfreeze only the last num_layers
            backbone_children = list(self.backbone.children())
            layers_to_unfreeze = backbone_children[-num_layers:]

            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True

        layers_msg = num_layers if num_layers else "all"
        print(f"Unfroze {layers_msg} backbone layers")


class SimpleEncoder(nn.Module):
    """
    Simple convolutional encoder for predicting firing rates from images.
    Uses a custom CNN architecture with batch normalization and pooling.
    """

    def __init__(self, out_neurons):
        super().__init__()
        # Deeper convolutional layers with batch normalization
        self.conv = nn.Sequential(
            # Initial conv layer with larger kernel to reduce spatial
            # dimensions
            nn.Conv2d(1, 64, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Middle conv layers
            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # Final pooling
            nn.AdaptiveAvgPool2d(1),
        )

        # Lightweight FC layers with single hidden layer
        self.fc = nn.Sequential(
            nn.Linear(512, 256),  # 131K parameters
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, out_neurons),  # 256K parameters for 1000 neurons
            nn.ELU(),
        )

    def forward(self, x):
        x = self.conv(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x + 1


class ResNetEncoder(nn.Module, ResNetMixin):
    """
    Neural encoder using ResNet18 backbone for predicting firing rates.
    Uses transfer learning with a pre-trained ResNet18 model.
    """

    def __init__(self, out_neurons, freeze_backbone=True):
        super().__init__()

        # Load pre-trained ResNet18
        self.backbone = models.resnet18(pretrained=True)
        feature_dim = 512

        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Add firing rate prediction head
        self.firing_head = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(512, out_neurons),
        )

        self.feature_dim = feature_dim

    def forward(self, x):
        # ResNet expects 3 channels, ensure input is correct
        if x.shape[1] == 1:  # If grayscale, repeat to make RGB
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {x.shape[1]}")

        # Extract features
        features = self.backbone(x).squeeze(-1).squeeze(-1)
        firing_rates = self.firing_head(features)
        return firing_rates + 1  # Ensure positive firing rates


class ResNetFromScratch(nn.Module, ResNetMixin):
    """
    ResNet18 encoder trained from scratch (no pre-trained weights).
    Uses the same architecture as ResNetEncoder but with random initialization.
    """

    def __init__(self, out_neurons, freeze_backbone=False):
        super().__init__()

        # Load ResNet18 architecture without pre-trained weights
        self.backbone = models.resnet18(pretrained=False)
        feature_dim = 512

        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Freeze backbone if requested (though typically not used for from-scratch)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Add firing rate prediction head
        self.firing_head = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(512, out_neurons),
        )

        self.feature_dim = feature_dim

    def forward(self, x):
        # ResNet expects 3 channels, ensure input is correct
        if x.shape[1] == 1:  # If grayscale, repeat to make RGB
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {x.shape[1]}")

        # Extract features
        features = self.backbone(x).squeeze(-1).squeeze(-1)
        firing_rates = self.firing_head(features)
        return firing_rates + 1  # Ensure positive firing rates


class ResNetConvOnly(nn.Module, ResNetMixin):
    """
    ResNet18 encoder using only pre-trained convolutional layers.
    Uses pre-trained ResNet18 conv backbone followed by a single linear layer
    for firing rate prediction. Simpler architecture than ResNetEncoder.
    """

    def __init__(self, out_neurons, freeze_backbone=False):
        super().__init__()

        # Load ResNet18 architecture WITHOUT pre-trained weights
        resnet = models.resnet18(pretrained=False)
        
        # Extract only the convolutional layers (remove avgpool and fc)
        # This gives us the pure conv feature extractor
        conv_layers = []
        for name, module in resnet.named_children():
            if name != "fc" and name != "avgpool":
                conv_layers.append(module)

        self.backbone = nn.Sequential(*conv_layers)
        
        # Feature dimension after layer4 is 512 channels
        feature_dim = 512

        # DON'T freeze backbone - we want to train it!
        # freeze_backbone parameter kept for compatibility but ignored
        
        # Single linear layer for firing rate prediction
        self.firing_head = nn.Linear(feature_dim, out_neurons)

        self.feature_dim = feature_dim

    def forward(self, x):
        # ResNet expects 3 channels, ensure input is correct
        if x.shape[1] == 1:  # If grayscale, repeat to make RGB
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {x.shape[1]}")

        # Extract features using conv layers only
        features = self.backbone(x)
        # Global average pooling to get fixed-size features
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.squeeze(-1).squeeze(-1)

        # Single linear layer prediction
        firing_rates = self.firing_head(features)
        return firing_rates + 1  # Ensure positive firing rates


class ResNetConv_2layerHead(nn.Module, ResNetMixin):
    """
    ResNet18 encoder using pre-trained convolutional layers with a 2-layer head.
    Uses pre-trained ResNet18 conv backbone followed by a two-layer MLP
    for firing rate prediction. Balance between capacity and simplicity,
    designed for better input optimization/reconstruction while maintaining
    good prediction performance.
    """

    def __init__(self, out_neurons, freeze_backbone=False):
        super().__init__()

        # Load ResNet18 architecture WITHOUT pre-trained weights
        resnet = models.resnet18(pretrained=False)
        
        # Extract only the convolutional layers (remove avgpool and fc)
        conv_layers = []
        for name, module in resnet.named_children():
            if name != "fc" and name != "avgpool":
                conv_layers.append(module)

        self.backbone = nn.Sequential(*conv_layers)
        
        # Feature dimension after layer4 is 512 channels
        feature_dim = 512
        
        # Two-layer MLP: 512 -> 256 -> neurons
        # Lighter regularization than ResNetEncoder for better reconstruction
        self.firing_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ELU(),
            nn.Dropout(0.1),  # Light dropout
            nn.Linear(256, out_neurons),
        )

        self.feature_dim = feature_dim

    def forward(self, x):
        # ResNet expects 3 channels, ensure input is correct
        if x.shape[1] == 1:  # If grayscale, repeat to make RGB
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {x.shape[1]}")

        # Extract features using conv layers only
        features = self.backbone(x)
        # Global average pooling to get fixed-size features
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)

        # Two-layer MLP prediction
        firing_rates = self.firing_head(features)
        return firing_rates + 1  # Ensure positive firing rates


class SimpleEncoderWithSkipConnection(nn.Module):
    """
    Simple convolutional encoder with skip connections for predicting firing
    rates from images.
    """

    def __init__(self, out_neurons):
        super().__init__()

        # Initial conv layer with larger kernel to reduce spatial dimensions
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Middle conv layers with skip connections
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # Skip connection projections to match dimensions
        self.skip1 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.skip2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.skip3 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)

        # Final pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

        # Lightweight FC layers with single hidden layer
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, out_neurons),
            nn.ELU(),
        )

    def forward(self, x):
        # Initial convolution
        x1 = self.conv1(x)

        # Middle convolutions with skip connections
        x2 = self.conv2(x1)
        x2 = x2 + self.skip1(x1)  # Skip connection 1

        x3 = self.conv3(x2)
        x3 = x3 + self.skip2(x2)  # Skip connection 2

        x4 = self.conv4(x3)
        x4 = x4 + self.skip3(x3)  # Skip connection 3

        # Final pooling and FC layers
        x = self.adaptive_pool(x4).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x + 1
