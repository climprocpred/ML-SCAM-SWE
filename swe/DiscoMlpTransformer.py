from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from torch_harmonics import DiscreteContinuousConvS2


@dataclass
class DiscoMlpTransformerConfig:
    in_channels: int
    mlp_out_dim: int
    disco_out_dim: int
    in_shape: Tuple[int, int]
    out_shape: Tuple[int, int]
    kernel_shape: Union[int, Tuple[int], Tuple[int, int]]
    basis_type: str = "piecewise linear"
    basis_norm_mode: str = "mean"
    groups: int = 1
    grid_in: str = "equiangular"
    grid_out: str = "equiangular"
    bias: bool = True
    disco_theta_cutoff: Optional[float] = None
    mlp_hidden_dim: Optional[int] = None
    nhead: int = 8
    num_layers: int = 4
    dropout: float = 0.0
    add_lat_sincos: bool = False  # Disabled by default since we use proper pos_embed
    disco_gate_init: float = 0.1
    # Spherical Transformer options
    attn_theta_cutoff: Optional[float] = None
    attention_mode: str = "neighborhood"
    norm_layer: str = "layer_norm"
    pos_embed: str = "spectral"
    grid_internal_pos: str = "legendre-gauss"  # often different from grid_out
    mlp_ratio: float = 2.0


from torch_harmonics.examples.models.s2transformer import SphericalAttentionBlock, SequencePositionEmbedding, SpectralPositionEmbedding, LearnablePositionEmbedding

class DiscoMlpTransformer(nn.Module):
    """
    Two-branch model:
      - MLP encoder on per-location channels
      - DiSCO convolution branch
    Branches are concatenated (no fusion MLP), with a learnable scalar gate
    applied to DiSCO (conservative init).
    
    Output is processed by a stack of SphericalAttentionBlocks (neighborhood attention).
    """

    def __init__(self, cfg: DiscoMlpTransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.in_shape = cfg.in_shape
        self.out_shape = cfg.out_shape

        if cfg.mlp_hidden_dim is None:
            cfg.mlp_hidden_dim = cfg.mlp_out_dim
        
        self.mlp_out_dim = cfg.mlp_out_dim
        self.disco_out_dim = cfg.disco_out_dim

        # MLP branch (per-location)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.in_channels, cfg.mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.mlp_hidden_dim, self.mlp_out_dim),
        )

        # DiSCO branch
        self.disco = DiscreteContinuousConvS2(
            in_channels=cfg.in_channels,
            out_channels=self.disco_out_dim,
            in_shape=cfg.in_shape,
            out_shape=cfg.out_shape,
            kernel_shape=cfg.kernel_shape,
            basis_type=cfg.basis_type,
            basis_norm_mode=cfg.basis_norm_mode,
            groups=cfg.groups,
            grid_in=cfg.grid_in,
            grid_out=cfg.grid_out,
            bias=cfg.bias,
            theta_cutoff=cfg.disco_theta_cutoff,
        )

        # Conservative DiSCO gate
        self.disco_gate = nn.Parameter(torch.tensor(cfg.disco_gate_init))

        # Lat sin/cos cache (for out_shape) - only used if add_lat_sincos is True
        if cfg.add_lat_sincos:
            nlat_out, nlon_out = cfg.out_shape
            lat = torch.linspace(-0.5 * torch.pi, 0.5 * torch.pi, nlat_out)
            self.register_buffer("_lat_sin", torch.sin(lat).view(1, nlat_out, 1, 1), persistent=False)
            self.register_buffer("_lat_cos", torch.cos(lat).view(1, nlat_out, 1, 1), persistent=False)

        # --- Spherical Transformer Setup ---
        
        extra_lat = 2 if cfg.add_lat_sincos else 0
        d_model = self.mlp_out_dim + self.disco_out_dim + extra_lat
        nlat_t, nlon_t = cfg.out_shape

        # 1. Positional Embedding
        if cfg.pos_embed == "sequence":
            self.pos_embed = SequencePositionEmbedding((nlat_t, nlon_t), num_chans=d_model, grid=cfg.grid_internal_pos)
        elif cfg.pos_embed == "spectral":
            self.pos_embed = SpectralPositionEmbedding((nlat_t, nlon_t), num_chans=d_model, grid=cfg.grid_internal_pos)
        elif cfg.pos_embed == "learnable lat":
            self.pos_embed = LearnablePositionEmbedding((nlat_t, nlon_t), num_chans=d_model, grid=cfg.grid_internal_pos, embed_type="lat")
        elif cfg.pos_embed == "learnable latlon":
            self.pos_embed = LearnablePositionEmbedding((nlat_t, nlon_t), num_chans=d_model, grid=cfg.grid_internal_pos, embed_type="latlon")
        elif cfg.pos_embed == "none":
            self.pos_embed = nn.Identity()
        else:
            raise ValueError(f"Unknown position embedding type {cfg.pos_embed}")

        # 2. Stack of Spherical Attention Blocks
        self.transformer = nn.Sequential(*[
            SphericalAttentionBlock(
                in_shape=cfg.out_shape,
                out_shape=cfg.out_shape,
                grid_in=cfg.grid_out,    # Input is on output grid of disco
                grid_out=cfg.grid_out,   # Output stays on same grid
                in_chans=d_model,
                out_chans=d_model,
                num_heads=cfg.nhead,
                mlp_ratio=cfg.mlp_ratio,
                drop_rate=cfg.dropout,
                drop_path=0.0, # Could expose drop_path_rate if needed
                act_layer=nn.GELU,
                norm_layer=cfg.norm_layer,
                use_mlp=True,
                bias=cfg.bias,
                attention_mode=cfg.attention_mode,
                theta_cutoff=cfg.attn_theta_cutoff,
            ) for _ in range(cfg.num_layers)
        ])

        # Decoder MLP
        # Matches Encoder: Linear(d_model_core -> hidden) -> GELU -> Linear(hidden -> in_channels)
        extra_lat = 2 if cfg.add_lat_sincos else 0
        d_model_core = self.mlp_out_dim + self.disco_out_dim + extra_lat
        self.decoder_mlp = nn.Sequential(
            nn.Linear(d_model_core, cfg.mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.mlp_hidden_dim, cfg.in_channels),
        )

    def _lat_sincos(self, device: torch.device, dtype: torch.dtype, nlon: int) -> torch.Tensor:
        lat_sin = self._lat_sin.to(device=device, dtype=dtype).expand(1, -1, nlon, 1)
        lat_cos = self._lat_cos.to(device=device, dtype=dtype).expand(1, -1, nlon, 1)
        return torch.cat([lat_sin, lat_cos], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        returns: [B, F, H, W] where F = 2*embed_dim (+2 if add_lat_sincos)
        """
        b, c, h, w = x.shape
        if (h, w) != self.out_shape:
            raise ValueError(
                f"Input spatial shape {(h, w)} must match out_shape {self.out_shape} "
                "for branch concat without resampling."
            )

        # MLP branch (per-location)
        x_mlp = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x_mlp = self.mlp(x_mlp)        # [B, H, W, E]

        # DiSCO branch
        x_disco = self.disco(x)        # [B, E, H, W]
        x_disco = (self.disco_gate * x_disco).permute(0, 2, 3, 1)  # [B, H, W, E]

        # Concatenate branches
        x_cat = torch.cat([x_mlp, x_disco], dim=-1)  # [B, H, W, 2E]

        # Append latitude sin/cos
        if self.cfg.add_lat_sincos:
            lat = self._lat_sincos(device=x.device, dtype=x.dtype, nlon=w)  # [1, H, W, 2]
            x_cat = torch.cat([x_cat, lat.expand(b, -1, -1, -1)], dim=-1) # [B, H, W, 2E+2]
        
        # Prepare for Spherical Transformer: Needs [B, C, H, W]
        out = x_cat.permute(0, 3, 1, 2).contiguous() # [B, F, H, W]

        # Apply Positional Embedding
        if isinstance(self.pos_embed, nn.Identity):
             pass # Skip
        else:
             # Pos embed typically expects [B, C, H, W]
             out = self.pos_embed(out)
        
        # Pass through Transformer Stack
        out = self.transformer(out) # [B, F, H, W]

        # Decoder Strategy:
        # 1. Permute to [B, H, W, F] for Linear layer
        out = out.permute(0, 2, 3, 1)

        # 2. Strip the last 'extra_lat' channels if they were added
        # if self.cfg.add_lat_sincos:
        #      lat_dims = 2
        #      out = out[..., :-lat_dims] # [B, H, W, d_model_core]

        # 3. Project back to input channels
        out = self.decoder_mlp(out) # [B, H, W, in_channels]
        
        # 4. Permute back to [B, C, H, W]
        out = out.permute(0, 3, 1, 2)

        return out