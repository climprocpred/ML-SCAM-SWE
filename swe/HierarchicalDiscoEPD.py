# coding=utf-8
"""
HierarchicalDiscoEPD: Hierarchical DiSCO encode -> process -> decode model.

This model uses stacked DiSCO convolutions to extract multi-scale spatial features:
- Level 1: Small FOV (~3x3 at equator) - local/CFL-scale features
- Level 2: Applied to Level 1 output - medium effective FOV
- Level 3: Applied to Level 2 output - larger effective FOV
- etc.

Key design choices:
- NO activation between DiSCO levels (preserves signed features for gradients)
- Kernel budget split across levels (not multiplied)
- Concatenate all levels' features for Head MLPs
- FFN for channel mixing as before
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import List, Optional, Tuple, Union
import numpy as np

from torch_harmonics import DiscreteContinuousConvS2, RealSHT, InverseRealSHT

# Import encoder/decoder and LatModulator from NNorm_GAM2 (reuse existing components)
from NNorm_GAM2 import Encoder, Decoder, LatModulator


class HierarchicalDiscoExtractor(nn.Module):
    """
    Extracts multi-scale spatial features using stacked DiSCO convolutions.
    
    Each level applies DiSCO to the previous level's output (no activation between).
    All levels' features are concatenated for downstream processing.
    """
    
    def __init__(
        self,
        num_levels: int,
        kernels_per_level: Union[int, List[int]],
        img_shape: Tuple[int, int],
        theta_cutoff: Union[float, List[float]],
        kernel_shape: Union[int, Tuple[int, int], List[Tuple[int, int]]] = (3, 3),
        basis_type: str = "piecewise linear",
        grid: str = "equiangular",
    ):
        """
        Args:
            num_levels: Number of hierarchical levels
            kernels_per_level: Kernels per level (int for equal, list for custom)
            img_shape: (n_lat, n_lon)
            theta_cutoff: Geodesic cutoff (float for same across levels, list for per-level)
            kernel_shape: DiSCO kernel shape (single for all levels, list for per-level)
            basis_type: DiSCO basis type
            grid: Grid type
        """
        super().__init__()
        
        self.num_levels = num_levels
        
        # Handle kernels_per_level as int or list
        if isinstance(kernels_per_level, int):
            self.kernels_per_level = [kernels_per_level] * num_levels
        else:
            assert len(kernels_per_level) == num_levels
            self.kernels_per_level = list(kernels_per_level)
        
        # Handle theta_cutoff as float or list
        if isinstance(theta_cutoff, (int, float)):
            self.theta_cutoffs = [float(theta_cutoff)] * num_levels
        else:
            assert len(theta_cutoff) == num_levels, f"theta_cutoff list length ({len(theta_cutoff)}) must match num_levels ({num_levels})"
            self.theta_cutoffs = list(theta_cutoff)
        
        # Handle kernel_shape as single value or list
        if isinstance(kernel_shape, (int, tuple)):
            self.kernel_shapes = [kernel_shape] * num_levels
        else:
            assert len(kernel_shape) == num_levels, f"kernel_shape list length ({len(kernel_shape)}) must match num_levels ({num_levels})"
            self.kernel_shapes = list(kernel_shape)
        
        self.total_kernels = sum(self.kernels_per_level)
        
        # Create DiSCO layers for each level
        # Level 1: in_channels=1 (applied per channel)
        # Level 2+: in_channels=kernels from previous level
        self.disco_layers = nn.ModuleList()
        
        for level_idx in range(num_levels):
            if level_idx == 0:
                in_ch = 1  # First level: applied to each channel independently
            else:
                in_ch = self.kernels_per_level[level_idx - 1]  # Previous level's output
            
            out_ch = self.kernels_per_level[level_idx]
            
            self.disco_layers.append(
                DiscreteContinuousConvS2(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    in_shape=img_shape,
                    out_shape=img_shape,
                    kernel_shape=self.kernel_shapes[level_idx],
                    basis_type=basis_type,
                    basis_norm_mode="mean",
                    groups=1,
                    grid_in=grid,
                    grid_out=grid,
                    bias=True,
                    theta_cutoff=self.theta_cutoffs[level_idx],
                )
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*C, 1, H, W) - single channel per batch element
        
        Returns:
            (B*C, total_kernels, H, W) - concatenated multi-scale features
        """
        level_outputs = []
        current = x  # (B*C, 1, H, W)
        
        for level_idx, disco in enumerate(self.disco_layers):
            # Apply DiSCO (no activation between levels!)
            current = disco(current)  # (B*C, kernels_this_level, H, W)
            
            # Store this level's output
            level_outputs.append(current)
        
        # Concatenate all levels: (B*C, total_kernels, H, W)
        return torch.cat(level_outputs, dim=1)


class HierarchicalDiscoMixerBlock(nn.Module):
    """
    Mixer block using hierarchical DiSCO for multi-scale spatial feature extraction.
    """
    
    def __init__(
        self,
        model_dim: int,
        n_heads: int,
        num_levels: int,
        kernels_per_level: Union[int, List[int]],
        head_mlp_hidden_dims: List[int],
        ffn_hidden_dims: List[int],
        img_shape: Tuple[int, int],
        theta_cutoff: Union[float, List[float]],
        kernel_shape: Union[int, Tuple[int, int], List[Tuple[int, int]]] = (3, 3),
        basis_type: str = "piecewise linear",
        grid: str = "equiangular",
        dropout_rate: float = 0.0,
        use_residual: bool = True,
        use_lat_modulation: bool = False,
    ):
        super().__init__()
        
        self.model_dim = model_dim
        self.learnable_dim = model_dim - 2  # Exclude sin/cos
        self.head_dim = self.learnable_dim // n_heads
        self.n_heads = n_heads
        self.use_residual = use_residual
        self.grad_ckpt_level = 0
        
        if self.learnable_dim % n_heads != 0:
            raise ValueError(f"learnable_dim ({self.learnable_dim}) must be divisible by n_heads ({n_heads})")
        
        # Hierarchical DiSCO extractor
        self.disco_extractor = HierarchicalDiscoExtractor(
            num_levels=num_levels,
            kernels_per_level=kernels_per_level,
            img_shape=img_shape,
            theta_cutoff=theta_cutoff,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            grid=grid,
        )
        
        total_kernels = self.disco_extractor.total_kernels
        
        # Per-head processing: takes total_kernels features -> 1 output
        self.head_processing = nn.ModuleList([
            self._build_head_mlp(total_kernels, head_mlp_hidden_dims)
            for _ in range(n_heads)
        ])
        
        # FFN for channel mixing
        ffn_layers = []
        current_dim = model_dim
        for h_dim in ffn_hidden_dims:
            ffn_layers.append(nn.Linear(current_dim, h_dim))
            ffn_layers.append(nn.GELU())
            current_dim = h_dim
        ffn_layers.append(nn.Linear(current_dim, self.learnable_dim))
        ffn_layers.append(nn.Dropout(dropout_rate))
        self.ffn = nn.Sequential(*ffn_layers)
        
        # Initialize FFN to ignore latitude channels initially
        with torch.no_grad():
            self.ffn[0].weight[:, -2:] = 0.0
        
        # Optional latitude modulation (FiLM)
        self.use_lat_modulation = use_lat_modulation
        if use_lat_modulation:
            # LatModulator: 2 (sin/cos) -> 2*total_kernels (gamma, beta)
            self.lat_modulator = LatModulator(total_kernels * 2, total_kernels * 2)
        else:
            self.lat_modulator = None
    
    def _build_head_mlp(self, in_dim: int, hidden_dims: List[int]) -> nn.Module:
        """Build MLP for head processing."""
        layers = []
        current_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.GELU())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, 1))
        return nn.Sequential(*layers)
    
    def set_grad_ckpt_level(self, level: int) -> None:
        self.grad_ckpt_level = level
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, model_dim)
        
        Returns:
            (B, H, W, model_dim)
        """
        B, H, W, D = x.shape
        
        # Separate learnable and immutable channels
        x_learn = x[..., :-2]  # (B, H, W, learnable_dim)
        sin_cos = x[..., -2:]  # (B, H, W, 2)
        
        x_res = x_learn
        
        # --- Hierarchical DiSCO Feature Extraction ---
        # Reshape: (B, H, W, C) -> (B, C, H, W) -> (B*C, 1, H, W)
        x_chw = x_learn.permute(0, 3, 1, 2)  # (B, C, H, W)
        x_flat = x_chw.reshape(B * self.learnable_dim, 1, H, W)
        
        # Extract hierarchical features
        if self.grad_ckpt_level >= 1 and self.training and x_flat.requires_grad:
            disco_out = checkpoint(self.disco_extractor, x_flat, use_reentrant=False)
        else:
            disco_out = self.disco_extractor(x_flat)  # (B*C, total_kernels, H, W)
        
        # Reshape: (B*C, K, H, W) -> (B, C, K, H, W) -> (B, H, W, C, K)
        total_kernels = disco_out.shape[1]
        disco_out = disco_out.view(B, self.learnable_dim, total_kernels, H, W)
        disco_out = disco_out.permute(0, 3, 4, 1, 2)  # (B, H, W, C, K)
        
        # --- FiLM Latitude Modulation (per-latitude-row) ---
        if self.use_lat_modulation and self.lat_modulator is not None:
            # Get sin/cos lat for each row: use first column (same across all columns)
            sin_cos_per_row = sin_cos[:, :, 0, :]  # (B, H, 2)
            
            # Compute gamma/beta: (B, H, 2) -> (B, H, 2*total_kernels)
            mod_params = self.lat_modulator(sin_cos_per_row)
            
            # Split into gamma/beta: (B, H, total_kernels) each
            gamma, beta = torch.chunk(mod_params, 2, dim=-1)
            
            # Reshape for broadcasting: (B, H, 1, 1, total_kernels)
            gamma = gamma.unsqueeze(2).unsqueeze(3)
            beta = beta.unsqueeze(2).unsqueeze(3)
            
            # Modulate: (B, H, W, C, K) * (B, H, 1, 1, K)
            disco_out = disco_out * gamma + beta
        
        # --- Per-Head Processing ---
        head_outputs = []
        for head_idx in range(self.n_heads):
            start_ch = head_idx * self.head_dim
            end_ch = start_ch + self.head_dim
            head_features = disco_out[..., start_ch:end_ch, :]  # (B, H, W, head_dim, K)
            
            # Flatten for MLP: (B*H*W*head_dim, K)
            head_flat = head_features.reshape(-1, total_kernels)
            
            # Apply head MLP
            if self.grad_ckpt_level >= 1 and self.training and head_flat.requires_grad:
                head_out = checkpoint(self.head_processing[head_idx], head_flat, use_reentrant=False)
            else:
                head_out = self.head_processing[head_idx](head_flat)  # (B*H*W*head_dim, 1)
            
            # Reshape: (B, H, W, head_dim)
            head_out = head_out.view(B, H, W, self.head_dim)
            head_outputs.append(head_out)
        
        x_learn = torch.cat(head_outputs, dim=-1)  # (B, H, W, learnable_dim)
        
        # First residual
        if self.use_residual:
            x_learn = x_learn + x_res
        
        # --- FFN Channel Mixing ---
        x_full = torch.cat([x_learn, sin_cos], dim=-1)
        x_res = x_learn
        x_learn = self.ffn(x_full)
        x_learn = x_learn + x_res  # Second residual
        
        return torch.cat([x_learn, sin_cos], dim=-1)


class HierarchicalDiscoEPD(nn.Module):
    """
    Hierarchical DiSCO encode -> process -> decode model.
    
    Uses stacked small-FOV DiSCO layers to build multi-scale features.
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (192, 288),
        in_chans: int = 3,
        out_chans: int = 3,
        model_dim: int = 66,
        n_heads: int = 8,
        num_mixer_blocks: int = 1,
        num_levels: int = 3,
        kernels_per_level: Union[int, List[int]] = 20,
        head_mlp_hidden_dims: List[int] = [32],
        ffn_hidden_dims: List[int] = [512, 256],
        encoder_hidden_dims: List[int] = [128],
        kernel_shape: Union[int, Tuple[int, int], List[Tuple[int, int]]] = (3, 3),
        theta_cutoff: Optional[Union[float, List[float]]] = None,
        basis_type: str = "piecewise linear",
        dropout_rate: float = 0.0,
        grad_ckpt_level: int = 1,
        grid: str = "equiangular",
        residual_prediction: bool = True,
        use_lat_modulation: bool = False,
        use_spectral_filter: bool = False,
        spectral_lmax: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        
        if kwargs:
            print(f"HierarchicalDiscoEPD: Ignoring unexpected kwargs: {list(kwargs.keys())}")
        
        self.n_lat, self.n_lon = img_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.M_features = in_chans + 2
        self.model_dim = model_dim
        self.grid_type = grid
        self.residual_prediction = residual_prediction
        self.grad_ckpt_level = grad_ckpt_level
        
        # Default theta_cutoff: ~3x3 at equator
        if theta_cutoff is None:
            theta_cutoff = (3 * torch.pi) / (torch.pi**0.5 * img_size[0])
        
        # --- Encoder and Decoder ---
        self.encoder = Encoder(self.M_features, model_dim, encoder_hidden_dims)
        self.decoder = Decoder(model_dim, self.M_features, encoder_hidden_dims[::-1])
        
        # --- Hierarchical DiSCO Mixer Blocks ---
        mixer_blocks = []
        for block_idx in range(num_mixer_blocks):
            mixer_blocks.append(
                HierarchicalDiscoMixerBlock(
                    model_dim=model_dim,
                    n_heads=n_heads,
                    num_levels=num_levels,
                    kernels_per_level=kernels_per_level,
                    head_mlp_hidden_dims=head_mlp_hidden_dims,
                    ffn_hidden_dims=ffn_hidden_dims,
                    img_shape=img_size,
                    theta_cutoff=theta_cutoff,
                    kernel_shape=kernel_shape,
                    basis_type=basis_type,
                    grid=grid,
                    dropout_rate=dropout_rate,
                    use_residual=(block_idx != 0),
                    use_lat_modulation=use_lat_modulation,
                )
            )
        self.mixer = nn.ModuleList(mixer_blocks)
        
        # --- Latitude Feature Buffer ---
        self.register_buffer('lat_features', None, persistent=False)
        
        # --- Spectral Filter (Optional) ---
        self.use_spectral_filter = use_spectral_filter
        if use_spectral_filter:
            lmax = spectral_lmax if spectral_lmax is not None else self.n_lat // 3
            mmax = lmax  # Isotropic filtering
            self.spectral_lmax = lmax
            self.sht = RealSHT(self.n_lat, self.n_lon, lmax=lmax, mmax=mmax, grid=grid)
            self.isht = InverseRealSHT(self.n_lat, self.n_lon, lmax=lmax, mmax=mmax, grid=grid)
            print(f"Spectral filter enabled: lmax={lmax} (removes modes > {lmax})")
        else:
            self.spectral_lmax = None
            self.sht = None
            self.isht = None
        
        # Set grad checkpoint levels
        self._set_grad_ckpt_levels(grad_ckpt_level)
        
        # Print parameter count
        self._print_param_counts()
    
    def _print_param_counts(self):
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        mixer_params = sum(p.numel() for p in self.mixer.parameters() if p.requires_grad)
        total_params = encoder_params + decoder_params + mixer_params
        
        print(f"HierarchicalDiscoEPD Model Parameters:")
        print(f"  Encoder:  {encoder_params:,}")
        print(f"  Mixer:    {mixer_params:,}")
        print(f"  Decoder:  {decoder_params:,}")
        print(f"  Total:    {total_params:,}")
    
    def _set_grad_ckpt_levels(self, level: int):
        self.grad_ckpt_level = level
        self.encoder.set_grad_ckpt_level(level)
        self.decoder.set_grad_ckpt_level(level)
        for block in self.mixer:
            if hasattr(block, "set_grad_ckpt_level"):
                block.set_grad_ckpt_level(level)
    
    def _get_lat_features(self, device, dtype) -> torch.Tensor:
        if self.lat_features is not None:
            if self.lat_features.device == device and self.lat_features.dtype == dtype:
                return self.lat_features
        
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
            lat = torch.linspace(-np.pi/2, np.pi/2, self.n_lat, device=device, dtype=dtype)
            sin_lat = torch.sin(lat)
            cos_lat = torch.cos(lat)
        
        sin_grid = sin_lat.view(-1, 1).expand(self.n_lat, self.n_lon)
        cos_grid = cos_lat.view(-1, 1).expand(self.n_lat, self.n_lon)
        features = torch.stack([sin_grid, cos_grid], dim=-1)
        
        self.lat_features = features
        return features
    
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        sin_cos_lat = x[..., -2:]
        encoded_learnable = self.encoder(x)
        return torch.cat([encoded_learnable, sin_cos_lat], dim=-1)
    
    def _process(self, latent_grid: torch.Tensor) -> torch.Tensor:
        x = latent_grid
        for block in self.mixer:
            if self.grad_ckpt_level >= 2 and self.training and x.requires_grad:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        delta = x - latent_grid
        delta = delta.clone()
        delta[..., -2:] = 0.0
        return delta
    
    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def _spectral_filter(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral lowpass filter to output.
        
        Removes high-frequency spherical harmonic modes (l > lmax).
        Input/output: (B, C, H, W)
        """
        if self.sht is None or self.isht is None:
            return x
        
        B, C, H, W = x.shape
        filtered_channels = []
        for c in range(C):
            # SHT expects (B, H, W), returns spectral coefficients
            spec = self.sht(x[:, c])
            # ISHT back to grid space (truncation happens automatically via lmax)
            filtered_channels.append(self.isht(spec))
        
        return torch.stack(filtered_channels, dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        if H != self.n_lat or W != self.n_lon:
            raise ValueError(f"Input dims ({H}, {W}) don't match model ({self.n_lat}, {self.n_lon})")
        
        # Prepare input
        x_perm = x.permute(0, 2, 3, 1)
        lat_feats = self._get_lat_features(x.device, x.dtype)
        lat_feats = lat_feats.unsqueeze(0).expand(B, -1, -1, -1)
        x_in = torch.cat([x_perm, lat_feats], dim=-1)
        
        # Encode
        latent = self._encode(x_in)
        
        # Process
        delta = self._process(latent)
        latent_next = latent + delta
        
        # Decode
        decoded = self._decode(latent_next)
        
        # Output
        out_phys = decoded[..., :-2]
        out = out_phys.permute(0, 3, 1, 2)
        
        if self.residual_prediction:
            out = out + x
        
        # Optional spectral lowpass filter
        if self.use_spectral_filter:
            out = self._spectral_filter(out)
        
        return out
