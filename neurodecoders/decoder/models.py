"""
Neural decoder model architectures.

This module contains neural network architectures for decoding images from
neural firing rates.
"""

import torch.nn as nn
import torch.nn.functional as F


class SimpleDecoder(nn.Module):
    """Fully connected + upsampling decoder to reconstruct images."""

    def __init__(self, in_neurons: int, image_size: int = 64):
        super().__init__()
        self.image_size = image_size

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
            nn.Linear(4096, 8 * 8 * 512),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 8, 8)
        x = self.deconv(x)
        x = F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return x
