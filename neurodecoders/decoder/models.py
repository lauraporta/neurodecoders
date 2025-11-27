"""
Neural decoder model architectures.

This module contains neural network architectures for decoding images from
neural firing rates.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDecoder(nn.Module):
    """Fully connected + upsampling decoder to reconstruct images."""

    def __init__(self, in_neurons: int, image_size: int = 64, output_scale: float = 2.5):
        """
        Args:
            in_neurons: Number of input neurons (neural activity dimension)
            image_size: Output image size
            output_scale: Scale factor for Tanh output to match z-scored image range.
                         Default 2.5 covers typical z-score range of ~[-2.5, 2.5]
        """
        super().__init__()
        self.image_size = image_size
        self.output_scale = output_scale

        self.fc = nn.Sequential(
            nn.Linear(in_neurons, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 8 * 8 * 128),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),  # Output [-1, 1], then scaled by output_scale
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, 8, 8)
        x = self.deconv(x)
        x = F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        # Scale to match z-scored image range
        return x * self.output_scale


class TransformerDecoder(nn.Module):
    """
    Transformer-based decoder that reconstructs images from neural activity.
    
    This architecture uses a Vision Transformer (ViT) style approach where:
    1. Neural activity is projected to a sequence of patch embeddings
    2. Transformer layers process these embeddings with self-attention
    3. Patch embeddings are reshaped and upsampled to form the output image
    
    Args:
        in_neurons: Number of input neurons (neural activity dimension)
        image_size: Output image size (must be divisible by patch_size)
        patch_size: Size of each image patch (default: 4)
        embed_dim: Dimension of transformer embeddings (default: 256)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 6)
        mlp_ratio: Ratio of MLP hidden dim to embed_dim (default: 4.0)
        dropout: Dropout rate (default: 0.1)
        output_scale: Scale factor for Tanh output to match z-scored image range (default: 2.5)
    """
    
    def __init__(
        self,
        in_neurons: int,
        image_size: int = 64,
        patch_size: int = 4,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        output_scale: float = 2.5,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.output_scale = output_scale
        
        # Number of patches in each dimension
        self.num_patches_side = image_size // patch_size
        self.num_patches = self.num_patches_side ** 2
        
        # Project neural activity to initial embedding
        self.neural_proj = nn.Sequential(
            nn.Linear(in_neurons, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        
        # Learnable patch embeddings (initialized from neural projection)
        self.patch_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )
        
        # Positional embeddings for patches
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # Final projection from embeddings to patch pixels
        pixels_per_patch = patch_size * patch_size
        self.patch_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, pixels_per_patch),
            nn.Tanh(),  # Output [-1, 1] to match normalized image range
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Project neural activity to conditioning vector
        neural_cond = self.neural_proj(x)  # (B, embed_dim)
        
        # Expand patch embeddings and add neural conditioning
        patches = self.patch_embed.expand(batch_size, -1, -1)  # (B, N, E)
        patches = patches + neural_cond.unsqueeze(1)  # Add conditioning to all patches
        
        # Add positional embeddings
        patches = patches + self.pos_embed
        
        # Apply transformer
        patches = self.transformer(patches)  # (B, N, E)
        
        # Project to pixel values
        patches = self.patch_proj(patches)  # (B, N, P*P)
        
        # Reshape to image
        patches = patches.view(
            batch_size,
            self.num_patches_side,
            self.num_patches_side,
            self.patch_size,
            self.patch_size,
        )
        # Rearrange: (B, H', W', P, P) -> (B, 1, H, W)
        x = patches.permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(batch_size, 1, self.image_size, self.image_size)
        
        # Scale to match z-scored image range
        return x * self.output_scale


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for diffusion timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalUNet(nn.Module):
    """
    U-Net architecture conditioned on neural activity and timestep.
    
    Used as the denoising network in the diffusion decoder.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4),
        neural_dim: int = 100,
        time_dim: int = 128,
    ):
        super().__init__()
        self.time_dim = time_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        
        # Neural activity embedding
        self.neural_mlp = nn.Sequential(
            nn.Linear(neural_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        
        # Encoder path
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        ch = base_channels
        in_ch = in_channels
        channels = [ch]
        
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.enc_blocks.append(
                self._make_block(in_ch, out_ch, time_dim)
            )
            self.downs.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            in_ch = out_ch
            channels.append(out_ch)
        
        # Bottleneck
        self.bottleneck = self._make_block(in_ch, in_ch, time_dim)
        
        # Decoder path
        self.dec_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.ups.append(nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1))
            skip_ch = channels.pop()
            self.dec_blocks.append(
                self._make_block(out_ch + skip_ch, out_ch, time_dim)
            )
            in_ch = out_ch
        
        # Output projection
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, in_channels, 3, padding=1),
        )
    
    def _make_block(self, in_ch: int, out_ch: int, time_dim: int) -> nn.Module:
        """Create a residual block with time/neural conditioning."""
        return ConditionalResBlock(in_ch, out_ch, time_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        neural_cond: torch.Tensor,
    ) -> torch.Tensor:
        # Get time and neural embeddings
        t_emb = self.time_mlp(t)
        n_emb = self.neural_mlp(neural_cond)
        cond = t_emb + n_emb  # Combined conditioning
        
        # Encoder
        skips = []
        for block, down in zip(self.enc_blocks, self.downs):
            x = block(x, cond)
            skips.append(x)
            x = down(x)
        
        # Bottleneck
        x = self.bottleneck(x, cond)
        
        # Decoder
        for block, up in zip(self.dec_blocks, self.ups):
            x = up(x)
            skip = skips.pop()
            # Handle potential size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x, cond)
        
        return self.out_conv(x)


class ConditionalResBlock(nn.Module):
    """Residual block with conditioning injection."""
    
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(min(8, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.cond_proj = nn.Linear(cond_dim, out_ch)
        self.conv2 = nn.Sequential(
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        # Add conditioning
        h = h + self.cond_proj(cond)[:, :, None, None]
        h = self.conv2(h)
        return h + self.skip(x)


class DiffusionDecoder(nn.Module):
    """
    Diffusion-based decoder that reconstructs images from neural activity.
    
    This architecture uses a denoising diffusion probabilistic model (DDPM)
    conditioned on neural activity to generate images. The model learns to
    denoise images starting from Gaussian noise, guided by the neural signal.
    
    During training, the model predicts the noise added to the image.
    During inference, iterative denoising generates the image from pure noise.
    
    Args:
        in_neurons: Number of input neurons (neural activity dimension)
        image_size: Output image size (default: 64)
        base_channels: Base number of channels in U-Net (default: 64)
        channel_mults: Channel multipliers for each U-Net level (default: (1, 2, 4))
        timesteps: Number of diffusion timesteps (default: 1000)
        beta_start: Starting beta for noise schedule (default: 1e-4)
        beta_end: Ending beta for noise schedule (default: 0.02)
        output_scale: Scale factor for output clipping to match z-scored image range (default: 2.5)
    """
    
    def __init__(
        self,
        in_neurons: int,
        image_size: int = 64,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4),
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        output_scale: float = 2.5,
    ):
        super().__init__()
        self.image_size = image_size
        self.timesteps = timesteps
        self.output_scale = output_scale
        
        # Denoising U-Net
        self.unet = ConditionalUNet(
            in_channels=1,
            base_channels=base_channels,
            channel_mults=channel_mults,
            neural_dim=in_neurons,
            time_dim=128,
        )
        
        # Setup noise schedule
        self.register_buffer("betas", torch.linspace(beta_start, beta_end, timesteps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer(
            "alphas_cumprod_prev",
            F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        )
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - self.alphas_cumprod)
        )
        self.register_buffer(
            "posterior_variance",
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward diffusion: add noise to image."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
    
    def get_loss(
        self,
        x_start: torch.Tensor,
        neural_cond: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the diffusion training loss.
        
        Args:
            x_start: Clean images (B, 1, H, W)
            neural_cond: Neural activity (B, N)
            t: Optional timesteps (B,), randomly sampled if not provided
        
        Returns:
            MSE loss between predicted and actual noise
        """
        batch_size = x_start.size(0)
        device = x_start.device
        
        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size,), device=device)
        
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        
        predicted_noise = self.unet(x_noisy, t.float(), neural_cond)
        
        return F.mse_loss(predicted_noise, noise)
    
    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: int,
        neural_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Single denoising step."""
        batch_size = x.size(0)
        device = x.device
        
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.float32)
        predicted_noise = self.unet(x, t_tensor, neural_cond)
        
        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        beta = self.betas[t]
        
        # Compute predicted x_0
        pred_x0 = (x - beta / torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha)
        
        # Compute mean
        if t > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(self.posterior_variance[t])
            return pred_x0 + sigma * noise
        else:
            return pred_x0
    
    @torch.no_grad()
    def sample(
        self,
        neural_cond: torch.Tensor,
        num_inference_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate images from neural activity via iterative denoising.
        
        Args:
            neural_cond: Neural activity (B, N)
            num_inference_steps: Number of denoising steps (default: self.timesteps)
        
        Returns:
            Generated images (B, 1, H, W)
        """
        batch_size = neural_cond.size(0)
        device = neural_cond.device
        
        if num_inference_steps is None:
            num_inference_steps = self.timesteps
        
        # Start from pure noise
        x = torch.randn(batch_size, 1, self.image_size, self.image_size, device=device)
        
        # Subsample timesteps if using fewer steps
        step_size = self.timesteps // num_inference_steps
        timesteps = list(range(0, self.timesteps, step_size))[::-1]
        
        for t in timesteps:
            x = self.p_sample(x, t, neural_cond)
        
        # Clamp to valid range matching z-scored image range
        return torch.clamp(x, -self.output_scale, self.output_scale)
    
    def forward(
        self,
        neural_cond: torch.Tensor,
        target_images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training or inference.
        
        During training (target_images provided): returns the diffusion loss
        During inference (no target_images): returns sampled images
        
        Args:
            neural_cond: Neural activity (B, N)
            target_images: Optional target images for training (B, 1, H, W)
        
        Returns:
            Loss tensor during training, generated images during inference
        """
        if target_images is not None:
            # Training mode: return loss
            return self.get_loss(target_images, neural_cond)
        else:
            # Inference mode: generate samples
            return self.sample(neural_cond)


def get_decoder_model(
    model_type: str,
    in_neurons: int,
    image_size: int = 32,
    **kwargs
) -> nn.Module:
    """
    Factory function to create decoder models with consistent interface.
    
    Args:
        model_type: Type of decoder model ('simple', 'transformer', 'diffusion')
        in_neurons: Number of input neurons
        image_size: Output image size (default: 32 for CIFAR-10)
        **kwargs: Additional model-specific parameters
            For 'transformer':
                - patch_size: Size of image patches (default: 4)
                - embed_dim: Transformer embedding dimension (default: 256)
                - num_heads: Number of attention heads (default: 8)
                - num_layers: Number of transformer layers (default: 6)
                - mlp_ratio: MLP hidden dim ratio (default: 4.0)
                - dropout: Dropout rate (default: 0.1)
            For 'diffusion':
                - base_channels: Base U-Net channels (default: 64)
                - channel_mults: Channel multipliers (default: (1, 2, 4))
                - timesteps: Diffusion timesteps (default: 1000)
                - beta_start: Noise schedule start (default: 1e-4)
                - beta_end: Noise schedule end (default: 0.02)
    
    Returns:
        Initialized decoder model
    
    Example:
        model = get_decoder_model('simple', in_neurons=100, image_size=32)
        model = get_decoder_model('transformer', in_neurons=100, num_layers=4)
        model = get_decoder_model('diffusion', in_neurons=100, timesteps=500)
    """
    model_type = model_type.lower()
    
    if model_type == "simple":
        return SimpleDecoder(
            in_neurons=in_neurons,
            image_size=image_size,
            output_scale=kwargs.get("output_scale", 2.5),
        )
    elif model_type == "transformer":
        return TransformerDecoder(
            in_neurons=in_neurons,
            image_size=image_size,
            patch_size=kwargs.get("patch_size", 4),
            embed_dim=kwargs.get("embed_dim", 256),
            num_heads=kwargs.get("num_heads", 8),
            num_layers=kwargs.get("num_layers", 6),
            mlp_ratio=kwargs.get("mlp_ratio", 4.0),
            dropout=kwargs.get("dropout", 0.1),
            output_scale=kwargs.get("output_scale", 2.5),
        )
    elif model_type == "diffusion":
        return DiffusionDecoder(
            in_neurons=in_neurons,
            image_size=image_size,
            base_channels=kwargs.get("base_channels", 64),
            channel_mults=kwargs.get("channel_mults", (1, 2, 4)),
            timesteps=kwargs.get("timesteps", 1000),
            beta_start=kwargs.get("beta_start", 1e-4),
            beta_end=kwargs.get("beta_end", 0.02),
            output_scale=kwargs.get("output_scale", 2.5),
        )
    else:
        raise ValueError(
            f"Unknown decoder model type: {model_type}. "
            f"Supported types: 'simple', 'transformer', 'diffusion'"
        )
