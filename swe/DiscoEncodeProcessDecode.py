# coding=utf-8
"""
DiscoEncodeProcessDecode: DiSCO-based encode -> process -> decode model.

This model replaces the naive patch-based spatial mixing with geodesically-aware
DiSCO convolutions. The architecture:

1. Encode: Physical state -> Latent state (pointwise MLP)
2. Process: DiSCO spatial mixing + head MLPs + FFN channel mixing
3. Decode: Latent state -> Physical state (pointwise MLP)

Key differences from EncodeProcessDecode:
- Uses DiSCO convolutions instead of patch extraction + linear
- No patching/padding needed (DiSCO handles spherical geometry)
- theta_cutoff controls receptive field instead of patch_size
- No FiLM modulation (DiSCO is geodesically aware, sin/cos lat in data)
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


class HeadProcessing(nn.Module):
    """
    Per-head processing after DiSCO spatial mixing.
    Takes DiSCO output features and applies head-specific transformations.
    """
    def __init__(self, in_dim: int, hidden_dims: List[int], output_dim: int, use_spectral_norm: bool = False):
        super().__init__()
        layers = []
        current_dim = in_dim
        for h_dim in hidden_dims:
            linear = nn.Linear(current_dim, h_dim)
            if use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.append(linear)
            layers.append(nn.GELU())
            current_dim = h_dim
        
        last_linear = nn.Linear(current_dim, output_dim)
        if use_spectral_norm:
            last_linear = nn.utils.spectral_norm(last_linear)
        layers.append(last_linear)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class DiscoMixerBlock(nn.Module):
    """
    DiSCO-based mixer block with:
    1. DiSCO spatial mixing (geodesically-aware convolutions)
    2. Per-head MLP processing
    3. FFN for channel mixing
    
    The DiSCO convolution learns shared_hidden kernels that are applied
    independently to each channel (depthwise style with shared kernels).
    """
    def __init__(
        self,
        model_dim: int,
        n_heads: int,
        shared_hidden: int,
        head_mlp_hidden_dims: List[int],
        ffn_hidden_dims: List[int],
        img_shape: Tuple[int, int],
        kernel_shape: Union[int, Tuple[int, int]] = (5, 5),
        theta_cutoff: Optional[float] = None,
        basis_type: str = "piecewise linear",
        grid: str = "equiangular",
        dropout_rate: float = 0.0,
        use_residual: bool = True,
        use_lat_modulation: bool = False,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        
        # model_dim includes sin/cos lat (last 2 channels are immutable)
        self.model_dim = model_dim
        self.learnable_dim = model_dim - 2
        self.head_dim = self.learnable_dim // n_heads
        self.n_heads = n_heads
        self.shared_hidden = shared_hidden
        self.use_residual = use_residual
        self.grad_ckpt_level = 0
        
        if self.learnable_dim % n_heads != 0:
            raise ValueError(f"learnable_dim ({self.learnable_dim}) must be divisible by n_heads ({n_heads})")
        
        # DiSCO convolution: learns shared_hidden kernels
        # Applied to each channel independently via reshape trick
        # in_channels=1, out_channels=shared_hidden means same kernels for all channels
        self.disco = DiscreteContinuousConvS2(
            in_channels=1,
            out_channels=shared_hidden,
            in_shape=img_shape,
            out_shape=img_shape,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            basis_norm_mode="mean",
            groups=1,
            grid_in=grid,
            grid_out=grid,
            bias=True,
            theta_cutoff=theta_cutoff,
        )
        
        # Per-head processing: takes shared_hidden features -> 1 output (for central prediction)
        # This mirrors the original head_processing structure
        self.head_processing = nn.ModuleList([
            HeadProcessing(shared_hidden, head_mlp_hidden_dims, 1, use_spectral_norm=use_spectral_norm)
            for _ in range(n_heads)
        ])
        
        # FFN for channel mixing (takes full model_dim as input for latitude awareness)
        ffn_layers = []
        current_dim = model_dim
        for h_dim in ffn_hidden_dims:
            linear = nn.Linear(current_dim, h_dim)
            if use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            ffn_layers.append(linear)
            ffn_layers.append(nn.GELU())
            current_dim = h_dim
        
        last_linear = nn.Linear(current_dim, self.learnable_dim)
        if use_spectral_norm:
            last_linear = nn.utils.spectral_norm(last_linear)
        ffn_layers.append(last_linear)
        ffn_layers.append(nn.Dropout(dropout_rate))
        self.ffn = nn.Sequential(*ffn_layers)
        
        # Initialize FFN to ignore latitude channels initially
        with torch.no_grad():
            self.ffn[0].weight[:, -2:] = 0.0
        
        # Optional latitude modulation (FiLM)
        self.use_lat_modulation = use_lat_modulation
        if use_lat_modulation:
            # LatModulator: 2 (sin/cos) -> 2*shared_hidden (gamma, beta)
            self.lat_modulator = LatModulator(shared_hidden * 2, shared_hidden * 2)
        else:
            self.lat_modulator = None
    
    def set_grad_ckpt_level(self, level: int) -> None:
        self.grad_ckpt_level = level
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, model_dim) - latent grid with sin/cos lat as last 2 channels
        
        Returns:
            (B, H, W, model_dim) - processed latent grid
        """
        B, H, W, D = x.shape
        
        # Separate learnable and immutable channels
        x_learn = x[..., :-2]  # (B, H, W, learnable_dim)
        sin_cos = x[..., -2:]  # (B, H, W, 2)
        
        # Save for residual
        x_res = x_learn
        
        # --- DiSCO Spatial Mixing ---
        # Reshape to apply same kernels to each channel: (B, C, H, W) -> (B*C, 1, H, W)
        x_chw = x_learn.permute(0, 3, 1, 2)  # (B, learnable_dim, H, W)
        x_flat = x_chw.reshape(B * self.learnable_dim, 1, H, W)
        
        # Apply DiSCO: (B*C, 1, H, W) -> (B*C, shared_hidden, H, W)
        if self.grad_ckpt_level >= 1 and self.training and x_flat.requires_grad:
            disco_out = checkpoint(self.disco, x_flat, use_reentrant=False)
        else:
            disco_out = self.disco(x_flat)
        
        # Reshape: (B*C, shared_hidden, H, W) -> (B, C, shared_hidden, H, W)
        disco_out = disco_out.view(B, self.learnable_dim, self.shared_hidden, H, W)
        
        # Permute to (B, H, W, C, shared_hidden) for head processing
        disco_out = disco_out.permute(0, 3, 4, 1, 2)  # (B, H, W, learnable_dim, shared_hidden)
        
        # --- FiLM Latitude Modulation (per-latitude-row) ---
        if self.use_lat_modulation:
            # Get sin/cos lat for each row: use first column (same across all columns)
            sin_cos_per_row = sin_cos[:, :, 0, :]  # (B, H, 2)
            
            # Compute gamma/beta: (B, H, 2) -> (B, H, 2*shared_hidden)
            mod_params = self.lat_modulator(sin_cos_per_row)
            
            # Split into gamma/beta: (B, H, shared_hidden) each
            gamma, beta = torch.chunk(mod_params, 2, dim=-1)
            
            # Reshape for broadcasting: (B, H, 1, 1, shared_hidden)
            gamma = gamma.unsqueeze(2).unsqueeze(3)
            beta = beta.unsqueeze(2).unsqueeze(3)
            
            # Modulate: (B, H, W, C, shared_hidden) * (B, H, 1, 1, shared_hidden)
            disco_out = disco_out * gamma + beta
        
        # --- Per-Head Processing ---
        # Split by heads: learnable_dim -> n_heads chunks of head_dim
        head_outputs = []
        for head_idx in range(self.n_heads):
            # Extract this head's channels: (B, H, W, head_dim, shared_hidden)
            start_ch = head_idx * self.head_dim
            end_ch = start_ch + self.head_dim
            head_features = disco_out[..., start_ch:end_ch, :]  # (B, H, W, head_dim, shared_hidden)
            
            # Flatten spatial for processing: (B*H*W*head_dim, shared_hidden)
            head_flat = head_features.reshape(-1, self.shared_hidden)
            
            # Apply head MLP: -> (B*H*W*head_dim, 1)
            if self.grad_ckpt_level >= 1 and self.training and head_flat.requires_grad:
                head_out = checkpoint(self.head_processing[head_idx], head_flat, use_reentrant=False)
            else:
                head_out = self.head_processing[head_idx](head_flat)
            
            # Reshape back: (B, H, W, head_dim)
            head_out = head_out.view(B, H, W, self.head_dim)
            head_outputs.append(head_out)
        
        # Concatenate heads: (B, H, W, learnable_dim)
        x_learn = torch.cat(head_outputs, dim=-1)
        
        # First residual connection
        if self.use_residual:
            x_learn = x_learn + x_res
        
        # --- FFN Channel Mixing ---
        # Include sin/cos for latitude awareness
        x_full = torch.cat([x_learn, sin_cos], dim=-1)  # (B, H, W, model_dim)
        
        x_res = x_learn
        x_learn = self.ffn(x_full)  # (B, H, W, learnable_dim)
        x_learn = x_learn + x_res  # Second residual
        
        # Re-append immutable sin/cos
        return torch.cat([x_learn, sin_cos], dim=-1)


class DiscoEncodeProcessDecode(nn.Module):
    """
    DiSCO-based encode -> process -> decode model.
    
    Uses geodesically-aware DiSCO convolutions for spatial mixing,
    replacing the naive patch-based approach.
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (256, 512),
        in_chans: int = 3,
        out_chans: int = 3,
        model_dim: int = 34,
        n_heads: int = 4,
        num_mixer_blocks: int = 4,
        shared_hidden: int = 64,
        head_mlp_hidden_dims: List[int] = [32],
        ffn_hidden_dims: List[int] = [512, 256],
        encoder_hidden_dims: List[int] = [256],
        kernel_shape: Union[int, Tuple[int, int]] = (5, 5),
        theta_cutoff: Optional[float] = None,
        basis_type: str = "piecewise linear",
        dropout_rate: float = 0.0,
        grad_ckpt_level: int = 1,
        grid: str = "equiangular",
        residual_prediction: bool = True,
        use_lat_modulation: bool = False,
        use_spectral_filter: bool = False,
        spectral_lmax: Optional[int] = None,
        use_hyperdiffusion: bool = False,
        hyperdiff_dt: float = 3600.0,
        hyperdiff_order: int = 4,
        n_history: int = 1,  # Number of input timesteps (1=single-step, 3=AB3-like)
        enforce_conservation: bool = False,  # If True, subtract global mean from residual
        use_spectral_norm: bool = False,
        **kwargs
    ):
        """
        Args:
            img_size: (n_lat, n_lon) tuple for grid dimensions
            in_chans: Number of input physical channels
            out_chans: Number of output physical channels
            model_dim: Latent space dimension (includes +2 for sin/cos lat)
            n_heads: Number of heads for spatial mixing
            num_mixer_blocks: Number of DiSCO mixer blocks
            shared_hidden: Number of DiSCO kernels (features per channel)
            head_mlp_hidden_dims: Hidden dimensions for head MLPs
            ffn_hidden_dims: Hidden dimensions for channel FFN
            encoder_hidden_dims: Hidden dimensions for encoder MLP
            kernel_shape: DiSCO kernel dimensions
            theta_cutoff: Geodesic cutoff radius (radians), None for default
            dropout_rate: Dropout rate in FFN
            grad_ckpt_level: Gradient checkpointing level
            grid: Grid type ("equiangular" or "legendre-gauss")
            residual_prediction: If True, model predicts residual
            use_lat_modulation: If True, apply FiLM modulation to DiSCO features
            use_spectral_filter: If True, apply spectral lowpass filter at output
            spectral_lmax: Max spherical harmonic degree for filter (default: nlat//3)
            use_hyperdiffusion: If True, apply hyperdiffusion in spectral space
            hyperdiff_dt: Timestep for hyperdiffusion damping (seconds), matches SWE solver
            hyperdiff_order: Power for hyperdiffusion (default 4 for 4th-order)
            n_history: Number of input timesteps expected (for multi-history models)
            enforce_conservation: If True, subtract global mean from residual to guarantee conservation
            use_spectral_norm: If True, apply spectral normalization to linear layers in mixer blocks
        """
        super().__init__()
        
        if kwargs:
            print(f"DiscoEncodeProcessDecode: Ignoring unexpected kwargs: {list(kwargs.keys())}")
        
        self.n_lat, self.n_lon = img_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.M_features = in_chans + 2  # Physical channels + sin/cos latitude
        self.model_dim = model_dim
        self.grid_type = grid
        self.residual_prediction = residual_prediction
        self.grad_ckpt_level = grad_ckpt_level
        self.n_history = n_history  # Store for training script to read
        
        if num_mixer_blocks < 1:
            raise ValueError("num_mixer_blocks must be at least 1.")
        
        # --- Encoder and Decoder ---
        self.encoder = Encoder(self.M_features, model_dim, encoder_hidden_dims)
        self.decoder = Decoder(model_dim, self.M_features, encoder_hidden_dims[::-1])
        
        # --- DiSCO Mixer Blocks ---
        mixer_blocks = []
        for block_idx in range(num_mixer_blocks):
            mixer_blocks.append(
                DiscoMixerBlock(
                    model_dim=model_dim,
                    n_heads=n_heads,
                    shared_hidden=shared_hidden,
                    head_mlp_hidden_dims=head_mlp_hidden_dims,
                    ffn_hidden_dims=ffn_hidden_dims,
                    img_shape=img_size,
                    kernel_shape=kernel_shape,
                    theta_cutoff=theta_cutoff,
                    basis_type=basis_type,
                    grid=grid,
                    dropout_rate=dropout_rate,
                    use_residual=(block_idx != 0),  # First block no residual
                    use_lat_modulation=use_lat_modulation,
                    use_spectral_norm=use_spectral_norm,
                )
            )
        self.mixer = nn.ModuleList(mixer_blocks)
        
        # --- Latitude Feature Buffer ---
        self.register_buffer('lat_features', None, persistent=False)
        
        # --- Spectral Filter and Hyperdiffusion (Optional) ---
        self.use_spectral_filter = use_spectral_filter
        self.use_hyperdiffusion = use_hyperdiffusion
        
        # Set up SHT/ISHT if either spectral filter or hyperdiffusion is enabled
        if use_spectral_filter or use_hyperdiffusion:
            lmax = spectral_lmax if spectral_lmax is not None else self.n_lat // 3
            mmax = lmax  # Isotropic filtering
            self.spectral_lmax = lmax
            self.sht = RealSHT(self.n_lat, self.n_lon, lmax=lmax, mmax=mmax, grid=grid)
            self.isht = InverseRealSHT(self.n_lat, self.n_lon, lmax=lmax, mmax=mmax, grid=grid)
            
            if use_spectral_filter:
                print(f"Spectral filter enabled: lmax={lmax} (hard truncation at l>{lmax})")
            
            # Hyperdiffusion setup (matches SWE solver behavior)
            if use_hyperdiffusion:
                # Compute Laplacian eigenvalues: -l(l+1) / R^2
                # For unit sphere (R=1), just use -l(l+1)
                l = torch.arange(0, lmax).reshape(lmax, 1).float()
                l = l.expand(lmax, mmax)
                lap = -l * (l + 1)
                
                # Hyperdiffusion: exp(-strength * (lap/lap_max)^order)
                # Matches SWE solver: exp((-dt/2/3600) * (lap/lap[-1,0])^4)
                strength = hyperdiff_dt / 2.0 / 3600.0
                normalized_lap = lap / lap[-1, 0]  # Range [0, 1]
                hyperdiff = torch.exp(-strength * normalized_lap ** hyperdiff_order)
                
                self.register_buffer('hyperdiff', hyperdiff)
                print(f"Hyperdiffusion enabled: dt={hyperdiff_dt}s, order={hyperdiff_order}, strength={strength:.6f}")
        else:
            self.spectral_lmax = None
            self.sht = None
            self.isht = None
            self.register_buffer('hyperdiff', None)
        
        # --- Conservation Constraint (Optional) ---
        self.enforce_conservation = enforce_conservation
        if enforce_conservation:
            # Compute quadrature weights for area-weighted global mean
            # For equiangular grid: weights are cos(lat)
            if grid == "equiangular":
                lat = torch.linspace(-np.pi/2, np.pi/2, self.n_lat)
                weights = torch.cos(lat)
            elif grid == "legendre-gauss":
                # For Gauss-Legendre, use the quadrature weights
                _, w = np.polynomial.legendre.leggauss(self.n_lat)
                weights = torch.tensor(w, dtype=torch.float32)
            else:
                lat = torch.linspace(-np.pi/2, np.pi/2, self.n_lat)
                weights = torch.cos(lat)
            
            # Normalize and register as buffer: (nlat,)
            weights = weights / weights.sum()
            self.register_buffer('conservation_weights', weights)
            print(f"Conservation constraint enabled: global mean subtracted from residual")
        else:
            self.register_buffer('conservation_weights', None)
        
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
        
        print(f"DiscoEncodeProcessDecode Model Parameters:")
        print(f"  Encoder:  {encoder_params:,}")
        print(f"  Mixer:    {mixer_params:,}")
        print(f"  Decoder:  {decoder_params:,}")
        print(f"  Total:    {total_params:,}")
    
    def _set_grad_ckpt_levels(self, level: int):
        """Set gradient checkpointing level for all components."""
        self.grad_ckpt_level = level
        self.encoder.set_grad_ckpt_level(level)
        self.decoder.set_grad_ckpt_level(level)
        for block in self.mixer:
            if hasattr(block, "set_grad_ckpt_level"):
                block.set_grad_ckpt_level(level)
    
    def _get_lat_features(self, device, dtype) -> torch.Tensor:
        """Generate or retrieve cached latitude features."""
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
        """Encode physical state to latent space."""
        sin_cos_lat = x[..., -2:]
        encoded_learnable = self.encoder(x)
        return torch.cat([encoded_learnable, sin_cos_lat], dim=-1)
    
    def _process(self, latent_grid: torch.Tensor) -> torch.Tensor:
        """Apply DiSCO mixer blocks to compute latent delta."""
        # Apply mixer blocks
        x = latent_grid
        for block in self.mixer:
            if self.grad_ckpt_level >= 2 and self.training and x.requires_grad:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        # Compute delta (difference from input)
        delta = x - latent_grid
        
        # Zero out sin/cos lat delta
        delta = delta.clone()
        delta[..., -2:] = 0.0
        
        return delta
    
    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent state back to physical space."""
        return self.decoder(z)
    
    def _spectral_filter(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral lowpass filter and/or hyperdiffusion to output.
        
        - If use_spectral_filter: Hard truncation at l > lmax (via SHT/ISHT)
        - If use_hyperdiffusion: Smooth exponential damping in spectral space
        - Both can be enabled independently or together
        
        Input/output: (B, C, H, W)
        """
        if self.sht is None or self.isht is None:
            return x
        
        B, C, H, W = x.shape
        filtered_channels = []
        for c in range(C):
            # SHT expects (B, H, W), returns spectral coefficients (B, lmax, mmax)
            spec = self.sht(x[:, c])
            
            # Apply hyperdiffusion in spectral space (element-wise multiply)
            if self.use_hyperdiffusion and self.hyperdiff is not None:
                spec = spec * self.hyperdiff
            
            # ISHT back to grid space (hard truncation happens automatically via lmax)
            filtered_channels.append(self.isht(spec))
        
        return torch.stack(filtered_channels, dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode -> process -> decode.
        
        Args:
            x: Input tensor of shape (B, C, H, W) where C = n_history * physical_chans
        
        Returns:
            Prediction tensor of shape (B, out_chans, H, W) - next single timestep
        """
        B, C, H, W = x.shape
        
        if H != self.n_lat or W != self.n_lon:
            raise ValueError(
                f"Input spatial dims ({H}, {W}) don't match model ({self.n_lat}, {self.n_lon})"
            )
        
        # For multi-history: extract most recent state for residual connection
        # Input is [oldest, ..., newest] stacked along channels
        if self.n_history > 1:
            physical_chans = C // self.n_history  # 3 for SWE
            x_recent = x[:, -physical_chans:, :, :]  # Last timestep (most recent)
        else:
            x_recent = x  # Single timestep input
        
        # 1. Prepare input: (B, C, H, W) -> (B, H, W, C+2)
        x_perm = x.permute(0, 2, 3, 1)
        lat_feats = self._get_lat_features(x.device, x.dtype)
        lat_feats = lat_feats.unsqueeze(0).expand(B, -1, -1, -1)
        x_in = torch.cat([x_perm, lat_feats], dim=-1)
        
        # 2. Encode
        latent = self._encode(x_in)
        
        # 3. Process (compute delta)
        delta = self._process(latent)
        
        # 4. Apply delta
        latent_next = latent + delta
        
        # 5. Decode
        decoded = self._decode(latent_next)
        
        # 6. Remove latitude features and permute back
        # For multi-history, take only out_chans (3) from decoded output
        if self.n_history > 1:
            out_phys = decoded[..., :self.out_chans]  # Take first out_chans features
        else:
            out_phys = decoded[..., :-2]  # Original behavior: remove sin/cos lat
        out = out_phys.permute(0, 3, 1, 2)
        
        # 7. Optional residual in physical space (use most recent timestep)
        if self.residual_prediction:
            # If conservation enabled, subtract area-weighted global mean from residual
            if self.enforce_conservation and self.conservation_weights is not None:
                # out is the raw residual (B, C, H, W) before adding to input
                # Compute area-weighted mean for each channel
                # weights: (nlat,) -> (1, 1, nlat, 1) for broadcasting
                w = self.conservation_weights.view(1, 1, -1, 1)
                # Weighted mean: sum over lat/lon, weighted by cos(lat)
                # (B, C, H, W) * (1, 1, H, 1) -> sum over W, weighted sum over H
                out_mean = (out * w).sum(dim=(2, 3), keepdim=True) / self.n_lon  # (B, C, 1, 1)
                out = out - out_mean  # Zero-mean residual
            out = out + x_recent
        
        # 8. Optional spectral lowpass filter and/or hyperdiffusion
        if self.use_spectral_filter or self.use_hyperdiffusion:
            out = self._spectral_filter(out)
        
        return out
