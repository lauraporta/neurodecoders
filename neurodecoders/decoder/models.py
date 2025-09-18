"""
Neural decoder model architectures.

This module contains neural network architectures for decoding images from
neural firing rates.
"""

import torch
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


class MirrorSimpleEncoderDecoder(nn.Module):
    """
    Architectural mirror of ``SimpleEncoder``.

    This decoder attempts to invert the ``SimpleEncoder`` pipeline by:
    - Reversing the final activation ``ELU`` and the ``+1`` shift
    - Mirroring the two FC layers back to a 512-dim feature vector
    - Expanding from a 1x1 feature map using transposed convolutions that
      mirror the encoder's convs and the max-pool (stride=2) stage

    Note: Exact mathematical inversion is impossible due to pooling, ReLU and
    dropout in the encoder. This model mirrors the architecture to provide a
    principled inverse structure.
    """

    def __init__(self, in_neurons: int, image_size: int = 64):
        super().__init__()
        self.image_size = image_size

        # Inverse of the encoder's two FC layers: out_neurons -> 256 -> 512
        self.fc_inv = nn.Sequential(
            nn.Linear(in_neurons, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )

        # Project 512 vector to 1x1 spatial feature, then mirror conv stack
        # Encoder order (after pool): 128(k7) -> 256(k5) -> 512(k3)
        # We invert this with transposed convolutions in reverse order.
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Invert the encoder's MaxPool2d(kernel=3, stride=2, padding=1)
            nn.ConvTranspose2d(
                64,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(),
            # Invert the first conv: Conv2d(1->64, k11, p5)
            nn.ConvTranspose2d(64, 1, kernel_size=11, stride=1, padding=5),
            nn.Tanh(),
        )

    @staticmethod
    def _inverse_elu_shift(x: torch.Tensor) -> torch.Tensor:
        """
        Approximate inverse of ``ELU`` with alpha=1 and the final ``+1`` shift
        done in the encoder: y = ELU(z) + 1.

        We invert as:
          t = y - 1
          z = t            if t >= 0
              ln(t + 1)    if t < 0
        """
        t = x - 1.0
        return torch.where(t >= 0.0, t, torch.log(t + 1.0))

    def forward(self, x):
        # Undo the encoder output transform (ELU + 1)
        x = self._inverse_elu_shift(x)

        # Mirror FC stack back to a 512-dim representation
        x = self.fc_inv(x)

        # Go to spatial 1x1 feature map and mirror conv/pool stack
        x = x.view(x.size(0), 512, 1, 1)
        x = self.deconv(x)

        # Final resize to requested image size
        x = F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return x
