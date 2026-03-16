# coding=utf-8
"""
EncodeProcessDecode: A simplified version of NNorm_GAM2.

This model follows a straightforward encode -> process -> decode pattern:
1. Encode: Physical state (B, H, W, M) -> Latent state (B, H, W, model_dim)
2. Process: Apply MLP-Mixer blocks in latent space to predict latent delta
3. Decode: Add delta and decode back to physical state

Unlike NNorm_GAM2, this model:
- Does NOT perform rollout in latent space (rollout happens in physical space outside)
- Does NOT compute multiple auxiliary training outputs
- Does NOT require a wrapper adapter for TH_SWE integration
- Has a simple forward(x) -> y interface compatible with (B, C, H, W) tensors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import List, Optional
import numpy as np

# Import the reusable components from NNorm_GAM2
from NNorm_GAM2 import (
    Encoder,
    Decoder,
    StandardMixerBlock,
    FinalMixerBlock,
    SpectralFilter,
)


class EncodeProcessDecode(nn.Module):
    """
    A simplified global atmospheric model with encode -> process -> decode structure.
    
    The model:
    1. Appends sin/cos latitude features to input
    2. Encodes to latent space
    3. Processes with MLP-Mixer blocks to get latent delta (residual)
    4. Adds delta to latent state and decodes back to physical space
    5. Strips latitude features and returns prediction
    
    Input/Output format: (B, C, H, W) for compatibility with torch-harmonics models.
    """
    
    def __init__(
        self,
        img_size: tuple = (256, 512),
        in_chans: int = 3,
        out_chans: int = 3,
        model_dim: int = 34,
        n_heads: int = 4,
        num_mixer_blocks: int = 4,
        head_mlp_hidden_dims: List[int] = [64, 32],
        ffn_hidden_dims: List[int] = [512, 256],
        encoder_hidden_dims: List[int] = [256],
        patch_size: int = 3,
        dropout_rate: float = 0.0,
        grad_ckpt_level: int = 1,
        grid: str = "equiangular",
        residual_prediction: bool = True,
        **kwargs
    ):
        """
        Args:
            img_size: (n_lat, n_lon) tuple for grid dimensions
            in_chans: Number of input physical channels
            out_chans: Number of output physical channels
            model_dim: Latent space dimension (includes +2 for sin/cos lat)
            n_heads: Number of heads for spatial mixing
            num_mixer_blocks: Number of mixer blocks in the processor
            head_mlp_hidden_dims: Hidden dimensions for spatial MLP heads
            ffn_hidden_dims: Hidden dimensions for channel FFN
            encoder_hidden_dims: Hidden dimensions for encoder MLP
            patch_size: Size of local patch for mixer (e.g., 3 for 3x3)
            dropout_rate: Dropout rate in FFN
            grad_ckpt_level: Gradient checkpointing level (0=off, 1=blocks, 2=full, 3=all)
            grid: Grid type ("equiangular" or "legendre-gauss")
            residual_prediction: If True, model predicts residual (added to input)
        """
        super().__init__()
        
        # Handle unexpected kwargs gracefully
        if kwargs:
            print(f"EncodeProcessDecode: Ignoring unexpected kwargs: {list(kwargs.keys())}")
        
        self.n_lat, self.n_lon = img_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.M_features = in_chans + 2  # Physical channels + sin/cos latitude
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.pad_size = patch_size // 2
        self.grid_type = grid
        self.residual_prediction = residual_prediction
        self.grad_ckpt_level = grad_ckpt_level
        
        if num_mixer_blocks < 1:
            raise ValueError("num_mixer_blocks must be at least 1.")
        
        # --- 1. Encoder and Decoder ---
        self.encoder = Encoder(self.M_features, model_dim, encoder_hidden_dims)
        self.decoder = Decoder(model_dim, self.M_features, encoder_hidden_dims[::-1])
        
        # --- 2. Mixer Blocks (Processor) ---
        mixer_blocks_list = []
        # Add N-1 Standard Mixer Blocks
        for block_idx in range(num_mixer_blocks - 1):
            mixer_blocks_list.append(
                StandardMixerBlock(
                    model_dim,
                    patch_size,
                    n_heads,
                    head_mlp_hidden_dims,
                    ffn_hidden_dims,
                    dropout_rate,
                    use_residual=(block_idx != 0)  # First block has no residual
                )
            )
        # Add the Final Mixer Block to collapse to central column
        mixer_blocks_list.append(
            FinalMixerBlock(
                model_dim, patch_size, n_heads,
                head_mlp_hidden_dims, ffn_hidden_dims, dropout_rate
            )
        )
        self.mixer = nn.ModuleList(mixer_blocks_list)
        
        # --- 3. Latitude Feature Buffer ---
        self.register_buffer('lat_features', None, persistent=False)
        
        # Set grad checkpoint levels
        self._set_grad_ckpt_levels(grad_ckpt_level)
        
        # Print parameter count
        self._print_param_counts()
    
    def _print_param_counts(self):
        """Print parameter counts for different components."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        mixer_params = sum(p.numel() for p in self.mixer.parameters() if p.requires_grad)
        total_params = encoder_params + decoder_params + mixer_params
        
        print(f"EncodeProcessDecode Model Parameters:")
        print(f"  Encoder:  {encoder_params:,}")
        print(f"  Mixer:    {mixer_params:,}")
        print(f"  Decoder:  {decoder_params:,}")
        print(f"  Total:    {total_params:,}")
    
    def _set_grad_ckpt_levels(self, level: int):
        """Set gradient checkpointing level for all components."""
        self.grad_ckpt_level = level
        
        # Encoder/Decoder use level 3 for their internal checkpointing
        self.encoder.set_grad_ckpt_level(level)
        self.decoder.set_grad_ckpt_level(level)
        
        # Mixer blocks use level 1/3 for block-level checkpointing
        block_level = 1 if level in (1, 3) else 0
        for block in self.mixer:
            if hasattr(block, "set_grad_ckpt_level"):
                block.set_grad_ckpt_level(block_level)
    
    def _get_lat_features(self, device, dtype) -> torch.Tensor:
        """Generate or retrieve cached latitude features (sin/cos)."""
        if self.lat_features is not None:
            if self.lat_features.device == device and self.lat_features.dtype == dtype:
                return self.lat_features
        
        # Generate latitude grid based on grid type
        if self.grid_type == "equiangular":
            lat = torch.linspace(-np.pi/2, np.pi/2, self.n_lat, device=device, dtype=dtype)
            sin_lat = torch.sin(lat)
            cos_lat = torch.cos(lat)
        elif self.grid_type == "legendre-gauss":
            try:
                cost, _ = np.polynomial.legendre.leggauss(self.n_lat)
            except AttributeError:
                cost = np.linspace(-1, 1, self.n_lat)
            sin_lat = torch.tensor(cost, device=device, dtype=dtype)
            cos_lat = torch.sqrt(1 - sin_lat**2)
        else:
            # Fallback to equiangular
            lat = torch.linspace(-np.pi/2, np.pi/2, self.n_lat, device=device, dtype=dtype)
            sin_lat = torch.sin(lat)
            cos_lat = torch.cos(lat)
        
        # Create 2D grid: (H, W, 2)
        sin_grid = sin_lat.view(-1, 1).expand(self.n_lat, self.n_lon)
        cos_grid = cos_lat.view(-1, 1).expand(self.n_lat, self.n_lon)
        features = torch.stack([sin_grid, cos_grid], dim=-1)
        
        self.lat_features = features
        return features
    
    def _apply_custom_padding(self, x: torch.Tensor) -> torch.Tensor:
        """Apply circular + polar padding for patch extraction."""
        # Input x shape: (B, H, W, C)
        # Circular padding for longitude
        x_circ = F.pad(
            x.permute(0, 3, 1, 2),
            (self.pad_size, self.pad_size, 0, 0),
            mode='circular'
        ).permute(0, 2, 3, 1)
        
        # Polar padding for latitude
        W = x_circ.shape[2]
        half_W = W // 2
        
        # North pole
        north_pole_rows = x_circ[:, :self.pad_size, :, :]
        north_pole_rows = torch.flip(north_pole_rows, dims=[1, 2])
        north_pole_rows = torch.roll(north_pole_rows, shifts=half_W, dims=2)
        
        # South pole
        south_pole_rows = x_circ[:, -self.pad_size:, :, :]
        south_pole_rows = torch.flip(south_pole_rows, dims=[1, 2])
        south_pole_rows = torch.roll(south_pole_rows, shifts=half_W, dims=2)
        
        x_padded = torch.cat([north_pole_rows, x_circ, south_pole_rows], dim=1)
        return x_padded
    
    def _extract_patches(self, grid_tensor: torch.Tensor) -> torch.Tensor:
        """Extract patches from grid for mixer input."""
        # Input: (B, H, W, C) -> Output: (B*H*W, patch_size^2, C)
        padded = self._apply_custom_padding(grid_tensor)
        padded_permuted = padded.permute(0, 3, 1, 2)
        
        num_patches = self.patch_size * self.patch_size
        feature_dim = grid_tensor.shape[-1]
        
        patches = padded_permuted.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, feature_dim, num_patches)
        patches = patches.permute(0, 2, 1)  # -> (B*H*W, num_patches, C)
        
        return patches
    
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode physical state to latent space."""
        # x: (B, H, W, M_features) with last 2 being sin/cos lat
        # Encoder outputs (B, H, W, model_dim - 2), then we append sin/cos
        sin_cos_lat = x[:, :, :, -2:]
        encoded_learnable = self.encoder(x)  # (B, H, W, model_dim - 2)
        return torch.cat([encoded_learnable, sin_cos_lat], dim=-1)  # (B, H, W, model_dim)
    
    def _process(self, latent_grid: torch.Tensor) -> torch.Tensor:
        """Apply mixer blocks to compute latent delta."""
        batch, height, width, _ = latent_grid.shape
        
        # Extract patches for mixer
        patches = self._extract_patches(latent_grid)  # (B*H*W, num_patches, model_dim)
        
        # Apply mixer blocks with optional checkpointing
        delta_flat = patches
        use_block_ckpt = self.grad_ckpt_level in (1, 3)
        
        for mixer_block in self.mixer:
            if use_block_ckpt and self.training and delta_flat.requires_grad:
                delta_flat = checkpoint(mixer_block, delta_flat, use_reentrant=False)
            else:
                delta_flat = mixer_block(delta_flat)
        
        # Reshape back to grid
        delta_out = delta_flat.view(batch, height, width, self.model_dim)
        
        # Zero out sin/cos lat delta to prevent drift
        delta_final = delta_out.clone()
        delta_final[..., -2:] = 0.0
        
        return delta_final
    
    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent state back to physical space."""
        # z: (B, H, W, model_dim)
        # Decoder outputs (B, H, W, M_features)
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode -> process -> decode.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Prediction tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Verify dimensions
        if H != self.n_lat or W != self.n_lon:
            raise ValueError(
                f"Input spatial dims ({H}, {W}) don't match model ({self.n_lat}, {self.n_lon})"
            )
        
        # 1. Prepare input: (B, C, H, W) -> (B, H, W, C+2)
        x_perm = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        
        # Get latitude features and concatenate
        lat_feats = self._get_lat_features(x.device, x.dtype)  # (H, W, 2)
        lat_feats = lat_feats.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
        x_in = torch.cat([x_perm, lat_feats], dim=-1)  # (B, H, W, C+2)
        
        # 2. Encode: (B, H, W, M) -> (B, H, W, model_dim)
        latent = self._encode(x_in)
        
        # 3. Process: compute latent delta
        delta = self._process(latent)
        
        # 4. Apply delta (residual in latent space)
        latent_next = latent + delta
        
        # 5. Decode: (B, H, W, model_dim) -> (B, H, W, M)
        decoded = self._decode(latent_next)
        
        # 6. Remove latitude features and permute back
        out_phys = decoded[..., :-2]  # (B, H, W, C)
        out = out_phys.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # 7. Optional residual prediction (in physical space)
        if self.residual_prediction:
            out = out + x
        
        return out
