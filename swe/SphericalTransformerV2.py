"""Enhanced Spherical Transformer model (V2) with global/local attention control.

This module provides an updated SphericalTransformer variant that adds
explicit global vs. neighborhood attention mode switching and other
architectural improvements over the torch-harmonics baseline.
"""

import torch
import torch.nn as nn


class SphericalTransformer(nn.Module):
    """Enhanced Spherical Transformer (V2) for atmospheric data on the sphere.

    An updated variant of the torch-harmonics SphericalTransformer with
    additional support for global context injection and improved positional
    embeddings on the sphere.

    Args:
        img_size: Spatial grid size as (nlat, nlon).
        in_chans: Number of input channels.
        out_chans: Number of output channels.
        embed_dim: Embedding dimension for transformer layers.
        num_layers: Number of transformer blocks.
        attention_mode: 'global' for full attention or 'neighborhood' for
            local windowed attention.
        residual_prediction: If True, model output is added to input.
        drop_path_rate: Stochastic depth probability.
        scale_factor: Downsampling scale factor for hierarchical models.
        grid: Spherical grid type ('equiangular' or 'legendre-gauss').
    """

    def __init__(self, img_size=(128, 256), in_chans=3, out_chans=3,
                 embed_dim=128, num_layers=4, attention_mode="global",
                 residual_prediction=True, drop_path_rate=0.0,
                 scale_factor=1, grid="equiangular", **kwargs):
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.attention_mode = attention_mode
        self.residual_prediction = residual_prediction

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, out_chans, H, W).
        """
        raise NotImplementedError
