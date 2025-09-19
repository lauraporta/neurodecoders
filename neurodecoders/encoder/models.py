"""
Neural encoder model architectures.

This module contains different neural network architectures for encoding
images to neural firing rates.
"""

import torch
import torch.nn as nn
import torchvision.models as models


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
    Uses pre-trained ResNet18 conv layers but replaces the fully connected
    layers with a custom firing rate prediction head.
    """

    def __init__(self, out_neurons, freeze_backbone=True):
        super().__init__()

        # Load pre-trained ResNet18
        self.backbone = models.resnet18(pretrained=True)
        feature_dim = 512

        # Extract only the convolutional layers (remove avgpool and fc)
        # This gives us the pure conv feature extractor
        conv_layers = []
        for name, module in self.backbone.named_children():
            if name != "fc" and name != "avgpool":
                conv_layers.append(module)

        self.backbone = nn.Sequential(*conv_layers)

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

        # Extract features using conv layers only
        features = self.backbone(x)
        # Global average pooling to get fixed-size features
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.squeeze(-1).squeeze(-1)

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
