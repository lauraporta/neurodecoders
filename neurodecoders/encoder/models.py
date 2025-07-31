"""
Neural encoder model architectures.

This module contains different neural network architectures for encoding
images to neural firing rates.
"""

import torch.nn as nn
import torchvision.models as models


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


class ResNetEncoder(nn.Module):
    """
    Neural encoder using ResNet backbone for predicting firing rates.
    Uses transfer learning with a pre-trained ResNet model.
    """

    def __init__(
        self, out_neurons, resnet_type="resnet18", freeze_backbone=True
    ):
        super().__init__()

        # Load pre-trained ResNet
        if resnet_type == "resnet18":
            self.backbone = models.resnet18(pretrained=True)
            feature_dim = 512
        elif resnet_type == "resnet34":
            self.backbone = models.resnet34(pretrained=True)
            feature_dim = 512
        elif resnet_type == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")

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

        self.resnet_type = resnet_type
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
