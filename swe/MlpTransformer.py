"""MLP-Transformer baseline model for spherical data.

This module provides a hybrid MLP + Transformer architecture that combines
per-location linear projections with global attention for spatial mixing
on spherical grids.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class MlpTransformerConfig:
    """Configuration for the MlpTransformer model.

    Attributes:
        model_dim: Hidden dimension / embedding size.
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        mlp_ratio: Ratio of MLP hidden dim to model_dim in transformer FFN.
        add_lat_sincos: If True, append sin/cos latitude features to input.
        dropout: Dropout probability.
    """
    model_dim: int = 64
    nhead: int = 2
    num_layers: int = 2
    mlp_ratio: float = 4.0
    add_lat_sincos: bool = True
    dropout: float = 0.0


class MlpTransformer(nn.Module):
    """Hybrid MLP + Spherical Transformer model.

    Combines a per-location MLP projection with a stack of global
    Transformer encoder layers for spatial mixing on latitude-longitude grids.

    Args:
        img_size: Spatial resolution as (nlat, nlon).
        in_chans: Number of input channels.
        out_chans: Number of output channels.
        residual_prediction: If True, model predicts a residual added to input.
        model_dim: Hidden embedding dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer layers.
        add_lat_sincos: Append sin/cos latitude positional features.
        drop_path_rate: Stochastic depth drop-path rate.
        grid: Grid type ('equiangular' or 'legendre-gauss').
    """

    def __init__(self, img_size=(128, 256), in_chans=3, out_chans=3,
                 residual_prediction=True, model_dim=64, nhead=2,
                 num_layers=2, add_lat_sincos=True, drop_path_rate=0.0,
                 grid="equiangular", **kwargs):
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.residual_prediction = residual_prediction

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, out_chans, H, W).
        """
        raise NotImplementedError
