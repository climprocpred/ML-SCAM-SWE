# coding=utf-8
"""
DISCO_NNorm_GAM2: Hybrid model combining DiSCO convolutions with NNorm_GAM2 training infrastructure.

This model combines:
- DiSCO geodesic convolutions (from DiscoEncodeProcessDecode) for spatial mixing
- Latent state processing infrastructure (from NNorm_LatentGlobalAtmosMixer2) for training

Key features:
1. Encoder: Pointwise MLP to go from physical features -> latent space
2. Process: DiSCO spatial mixing with FiLM latitude modulation + FFN channel mixing
3. Decoder: Pointwise MLP to go from latent -> physical
4. Full training infrastructure: autoregressive rollout, TBPTT, sub-stepping
5. Optional SHT-based spectral lowpass filter
6. Optional SFNO post-processor

Input/Output: (B, H, W, M_features) - matches NNorm_LatentGlobalAtmosMixer2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import List, Optional, Tuple, Union
import numpy as np
import hashlib
import pickle
from pathlib import Path
import time

from torch_harmonics import DiscreteContinuousConvS2, RealSHT, InverseRealSHT

# Import encoder/decoder and LatModulator from NNorm_GAM2 (reuse existing components)
import sys
sys.path.append('/glade/work/idavis/ml_scam/latent/train_encoder_mixer_decoder/')
from NNorm_GAM2 import Encoder, Decoder, LatModulator

# Try to import SFNO
try:
    from torch_harmonics.examples.sfno import SphericalFourierNeuralOperator
    SFNO_AVAILABLE = True
except ImportError:
    SFNO_AVAILABLE = False

# Default cache directory for DiSCO basis matrices
DISCO_CACHE_DIR = Path("/glade/work/idavis/ml_scam/latent/train_encoder_mixer_decoder/TH_SWE/.disco_cache")


def _get_disco_cache_key(
    in_channels: int,
    out_channels: int,
    in_shape: Tuple[int, int],
    out_shape: Tuple[int, int],
    kernel_shape: Union[int, Tuple[int, int]],
    basis_type: str,
    theta_cutoff: Optional[float],
    grid_in: str,
    grid_out: str,
) -> str:
    """Generate a unique cache key based on DiSCO configuration."""
    config_str = (
        f"in_channels={in_channels},"
        f"out_channels={out_channels},"
        f"in_shape={in_shape},"
        f"out_shape={out_shape},"
        f"kernel_shape={kernel_shape},"
        f"basis_type={basis_type},"
        f"theta_cutoff={theta_cutoff},"
        f"grid_in={grid_in},"
        f"grid_out={grid_out}"
    )
    return hashlib.md5(config_str.encode()).hexdigest()


def create_cached_disco_conv(
    in_channels: int,
    out_channels: int,
    in_shape: Tuple[int, int],
    out_shape: Tuple[int, int],
    kernel_shape: Union[int, Tuple[int, int]],
    basis_type: str,
    theta_cutoff: Optional[float],
    grid: str,
    cache_dir: Optional[Path] = None,
) -> DiscreteContinuousConvS2:
    """
    Create a DiscreteContinuousConvS2 with cached state.
    
    On first run: creates the layer (slow) and caches the full state_dict.
    On subsequent runs: loads state from cache (fast).
    
    Note: The first run for a given config will still be slow since we must
    compute the basis. But subsequent runs with the same config will load
    from cache almost instantly.
    """
    if cache_dir is None:
        cache_dir = DISCO_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_key = _get_disco_cache_key(
        in_channels, out_channels, in_shape, out_shape,
        kernel_shape, basis_type, theta_cutoff, grid, grid
    )
    cache_file = cache_dir / f"disco_state_{cache_key}.pt"
    
    # Common kwargs for layer creation
    disco_kwargs = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        in_shape=in_shape,
        out_shape=out_shape,
        kernel_shape=kernel_shape,
        basis_type=basis_type,
        basis_norm_mode="mean",
        groups=1,
        grid_in=grid,
        grid_out=grid,
        bias=True,
        theta_cutoff=theta_cutoff,
    )
    
    # Try to load from cache
    if cache_file.exists():
        try:
            # Load the cached state
            cached_state = torch.load(cache_file, map_location='cpu', weights_only=False)
            
            # Create a minimal disco layer - we'll replace everything from cache
            # Unfortunately, we still need to call the constructor to get the module structure
            # But we can at least provide a smaller dummy psi to speed things up
            disco = DiscreteContinuousConvS2(**disco_kwargs)
            
            # Load the cached state (this overwrites psi with the cached version)
            disco.load_state_dict(cached_state, strict=False)
            
            print(f"(loaded from cache)")
            return disco
            
        except Exception as e:
            print(f"(cache load failed: {e}, recomputing)")
    
    # Create fresh layer (this is the slow part - basis computation)
    disco = DiscreteContinuousConvS2(**disco_kwargs)
    
    # Cache the full state for future runs
    try:
        torch.save(disco.state_dict(), cache_file)
        print(f"(saved to cache)")
    except Exception as e:
        print(f"(cache save failed: {e})")
    
    return disco
    
    return disco


class HeadProcessing(nn.Module):
    """
    Per-head processing after DiSCO spatial mixing.
    Takes DiSCO output features and applies head-specific transformations.
    """
    def __init__(self, in_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers = []
        current_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.GELU())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class DiscoMixerBlock(nn.Module):
    """
    DiSCO-based mixer block with FiLM latitude modulation (always enabled).
    
    Components:
    1. DiSCO spatial mixing (geodesically-aware convolutions)
    2. FiLM latitude modulation on DiSCO features
    3. Per-head MLP processing
    4. FFN for channel mixing
    
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
        # Uses caching to avoid recomputing basis matrices on every initialization
        self.disco = create_cached_disco_conv(
            in_channels=1,
            out_channels=shared_hidden,
            in_shape=img_shape,
            out_shape=img_shape,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            theta_cutoff=theta_cutoff,
            grid=grid,
        )
        
        # Per-head processing: takes shared_hidden features -> 1 output (for central prediction)
        self.head_processing = nn.ModuleList([
            HeadProcessing(shared_hidden, head_mlp_hidden_dims, 1)
            for _ in range(n_heads)
        ])
        
        # FFN for channel mixing (takes full model_dim as input for latitude awareness)
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
        
        # FiLM Latitude Modulation (always enabled)
        # LatModulator: 2 (sin/cos) -> 2*shared_hidden (gamma, beta)
        self.lat_modulator = LatModulator(shared_hidden * 2, shared_hidden * 2)
    
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


class DISCO_NNorm_GAM2(nn.Module):
    """
    Hybrid model combining DiSCO convolutions with NNorm_GAM2 training infrastructure.
    
    Uses geodesically-aware DiSCO convolutions for spatial mixing, with the full
    training infrastructure from NNorm_LatentGlobalAtmosMixer2 (rollout, TBPTT, etc.).
    
    Input/Output: (B, H, W, M_features) - matches NNorm_LatentGlobalAtmosMixer2
    """
    
    def __init__(
        self,
        M_features: int,
        model_dim: int,
        n_heads: int,
        num_mixer_blocks: int,
        shared_hidden: int,
        head_mlp_hidden_dims: List[int],
        ffn_hidden_dims: List[int],
        encoder_hidden_dims: List[int],
        img_size: Tuple[int, int] = (256, 512),
        kernel_shape: Union[int, Tuple[int, int]] = (5, 5),
        theta_cutoff: Optional[float] = None,
        basis_type: str = "piecewise linear",
        grid: str = "equiangular",
        dropout_rate: float = 0.0,
        grad_ckpt_level: int = 1,
        teacher_forcing_requires_grad: bool = True,
        use_spectral_filter: bool = False,
        spectral_lmax: Optional[int] = None,
        use_sfno: bool = False,
        sfno_embed_dim: int = 64,
        sfno_num_layers: int = 3,
        sfno_use_mlp: bool = True,
        sfno_mlp_ratio: int = 4,
        sfno_scale_factor: int = 2,
        input_substeps: int = 1,
        compute_tendencies: bool = True,
        **kwargs
    ):
        """
        Args:
            M_features: Number of input physical channels (including sin/cos lat)
            model_dim: Latent space dimension (includes +2 for sin/cos lat)
            n_heads: Number of heads for spatial mixing
            num_mixer_blocks: Number of DiSCO mixer blocks
            shared_hidden: Number of DiSCO kernels (features per channel)
            head_mlp_hidden_dims: Hidden dimensions for head MLPs
            ffn_hidden_dims: Hidden dimensions for channel FFN
            encoder_hidden_dims: Hidden dimensions for encoder MLP
            img_size: (n_lat, n_lon) tuple for grid dimensions
            kernel_shape: DiSCO kernel dimensions
            theta_cutoff: Geodesic cutoff radius (radians), None for default
            basis_type: DiSCO basis type
            grid: Grid type ("equiangular" or "legendre-gauss")
            dropout_rate: Dropout rate in FFN
            grad_ckpt_level: Gradient checkpointing level (0-3)
            teacher_forcing_requires_grad: Whether to track gradients on teacher forcing
            use_spectral_filter: If True, apply SHT lowpass filter at output
            spectral_lmax: Max spherical harmonic degree for filter (default: nlat//3)
            use_sfno: If True, apply SFNO post-processor
            sfno_*: SFNO configuration parameters
            input_substeps: Number of sub-steps per forward pass
            compute_tendencies: If False, skip tendency decoder to save memory
        """
        super().__init__()
        
        if kwargs:
            print(f"DISCO_NNorm_GAM2: Ignoring unexpected kwargs: {list(kwargs.keys())}")
        
        self.n_lat, self.n_lon = img_size
        self.M_features = M_features
        self.model_dim = model_dim
        self.grid_type = grid
        self.grad_ckpt_level = grad_ckpt_level
        self.teacher_forcing_requires_grad = teacher_forcing_requires_grad
        self.input_substeps = input_substeps
        
        # Default values for training loop
        self.default_num_rollout_steps = 1
        self.default_decode_freq = 1
        self.is_training_mode = True
        self.default_compute_tendencies = compute_tendencies  # Set False to skip tendency decoding and save memory
        
        if num_mixer_blocks < 1:
            raise ValueError("num_mixer_blocks must be at least 1.")
        
        # --- Encoder and Decoder ---
        self.encoder = Encoder(M_features, model_dim, encoder_hidden_dims)
        self.decoder = Decoder(model_dim, M_features, encoder_hidden_dims[::-1])
        self.tend_decoder = Decoder(model_dim, M_features, encoder_hidden_dims[::-1])
        
        # --- DiSCO Mixer Blocks ---
        print(f"  Building {num_mixer_blocks} DiSCO mixer blocks (this may take a while for large grids)...")
        mixer_blocks = []
        for block_idx in range(num_mixer_blocks):
            print(f"    Initializing mixer block {block_idx + 1}/{num_mixer_blocks}...", end=" ", flush=True)
            import time
            start_time = time.time()
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
                )
            )
            elapsed = time.time() - start_time
            print(f"done ({elapsed:.1f}s)")
        self.mixer = nn.ModuleList(mixer_blocks)
        print(f"  All mixer blocks initialized.")
        
        # --- Latitude Feature Buffer ---
        self.register_buffer('lat_features', None, persistent=False)
        
        # --- Spectral Filter (SHT-based lowpass) ---
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
        
        # --- SFNO Post-Processor (Optional) ---
        self.use_sfno = use_sfno
        if use_sfno:
            if not SFNO_AVAILABLE:
                raise ImportError(
                    "torch_harmonics is required for SFNO. Install with: pip install torch_harmonics"
                )
            self.sfno = SphericalFourierNeuralOperator(
                img_size=(self.n_lat, self.n_lon),
                in_chans=model_dim,
                out_chans=model_dim,
                embed_dim=sfno_embed_dim,
                num_layers=sfno_num_layers,
                use_mlp=sfno_use_mlp,
                mlp_ratio=sfno_mlp_ratio,
                residual_prediction=True,
                scale_factor=sfno_scale_factor,
                grid=grid,
                normalization_layer="none",
                pos_embed="learnable lat"
            )
        else:
            self.sfno = None
        
        # Set grad checkpoint levels
        self._set_grad_ckpt_levels(grad_ckpt_level)
        
        # Print parameter count
        self._print_param_counts()
    
    def _print_param_counts(self):
        """Print parameter counts for different components."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        tend_decoder_params = sum(p.numel() for p in self.tend_decoder.parameters() if p.requires_grad)
        mixer_params = sum(p.numel() for p in self.mixer.parameters() if p.requires_grad)
        sfno_params = sum(p.numel() for p in self.sfno.parameters() if p.requires_grad) if self.sfno else 0
        total_params = encoder_params + decoder_params + tend_decoder_params + mixer_params + sfno_params
        
        print(f"DISCO_NNorm_GAM2 Model Parameters:")
        print(f"  Encoder:       {encoder_params:,}")
        print(f"  Mixer:         {mixer_params:,}")
        print(f"  Decoder:       {decoder_params:,}")
        print(f"  TendDecoder:   {tend_decoder_params:,}")
        if sfno_params:
            print(f"  SFNO:          {sfno_params:,}")
        print(f"  Total:         {total_params:,}")
    
    def _set_grad_ckpt_levels(self, level: int):
        """Set gradient checkpointing level for all components."""
        self.grad_ckpt_level = level
        self.encoder.set_grad_ckpt_level(level)
        self.decoder.set_grad_ckpt_level(level)
        self.tend_decoder.set_grad_ckpt_level(level)
        for block in self.mixer:
            if hasattr(block, "set_grad_ckpt_level"):
                block.set_grad_ckpt_level(1 if level in (1, 3) else 0)
    
    def set_grad_ckpt_level(self, level: int) -> None:
        if level not in (0, 1, 2, 3):
            raise ValueError("grad_ckpt_level must be one of {0, 1, 2, 3}.")
        self._set_grad_ckpt_levels(level)
    
    def set_teacher_forcing_grad(self, enabled: bool) -> None:
        self.teacher_forcing_requires_grad = bool(enabled)
    
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
    
    def _encode_global(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a global grid to latent space.
        
        Args:
            x: (B, H, W, M_features) - physical state grid
               Last 2 features are sin/cos of latitude (immutable)
        
        Returns:
            encoded: (B, H, W, model_dim) - latent state grid
                     Last 2 dims are sin/cos lat (immutable, copied from input)
        """
        # Extract immutable sin/cos latitude from input (last 2 features)
        sin_cos_lat = x[:, :, :, -2:]  # (B, H, W, 2)
        
        # Standard pointwise encoding (outputs learnable_dim)
        encoded_learnable = self.encoder(x)  # (B, H, W, model_dim - 2)
        
        # Append sin/cos lat to get full latent
        return torch.cat([encoded_learnable, sin_cos_lat], dim=-1)
    
    def _compute_latent_delta(self, latent_grid: torch.Tensor) -> torch.Tensor:
        """
        Apply DiSCO mixer blocks to compute latent delta.
        
        Args:
            latent_grid: (B, H, W, model_dim)
        
        Returns:
            delta: (B, H, W, model_dim) with last 2 channels zeroed
        """
        # Apply mixer blocks
        x = latent_grid
        use_block_ckpt = self.grad_ckpt_level in (1, 3)
        
        for block in self.mixer:
            if use_block_ckpt and self.training and x.requires_grad:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        # Compute delta (difference from input)
        delta = x - latent_grid
        
        # Zero out sin/cos lat delta (they are immutable)
        delta = delta.clone()
        delta[..., -2:] = 0.0
        
        return delta
    
    def _spectral_filter(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral lowpass filter to output.
        
        Removes high-frequency spherical harmonic modes (l > lmax).
        Input/output: (B, H, W, C) or (B, C, H, W)
        
        Note: Disables autocast and casts to float32 for SHT since cuFFT 
        in half precision only supports power-of-2 sizes.
        """
        if self.sht is None or self.isht is None:
            return x
        
        # Save original dtype for casting back
        original_dtype = x.dtype
        
        # Determine input format and convert if needed
        if x.shape[-1] == self.M_features or x.shape[-1] == 2:
            # Input is (B, H, W, C) - need to permute for SHT
            x_bhwc = True
            x = x.permute(0, 3, 1, 2)  # -> (B, C, H, W)
        else:
            x_bhwc = False
        
        B, C, H, W = x.shape
        
        # Cast to float32 for SHT (cuFFT limitation with non-power-of-2 in half precision)
        x_f32 = x.float()
        
        # Disable autocast to ensure SHT/ISHT run in float32
        device_type = 'cuda' if x.is_cuda else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=False):
            filtered_channels = []
            for c in range(C):
                # SHT expects (B, H, W), returns spectral coefficients
                spec = self.sht(x_f32[:, c])
                # ISHT back to grid space (truncation happens automatically via lmax)
                filtered_channels.append(self.isht(spec))
            
            filtered = torch.stack(filtered_channels, dim=1)
        
        # Cast back to original dtype
        filtered = filtered.to(original_dtype)
        
        if x_bhwc:
            filtered = filtered.permute(0, 2, 3, 1)  # -> (B, H, W, C)
        
        return filtered
    
    def forward(
        self,
        x_current_global: torch.Tensor,
        x_next_global: Optional[torch.Tensor] = None,
        *,
        num_rollout_steps: Optional[int] = None,
        decode_freq: Optional[int] = None,
        train: Optional[bool] = None,
        global_mode: Optional[bool] = True,
        TBPTT_latent_state: Optional[torch.Tensor] = None,
        TBPTT_prev_true_latent: Optional[torch.Tensor] = None,
        input_substeps: Optional[int] = None,
        compute_tendencies: Optional[bool] = None,
    ):
        """
        Forward pass with full training infrastructure.
        
        Args:
            x_current_global: (B, H, W, M_features) - current physical state
            x_next_global: (B, steps, H, W, M_features) - future states for training
            num_rollout_steps: Number of autoregressive steps
            decode_freq: Decode every N steps (sparse decoding for efficiency)
            train: Training mode flag
            global_mode: If True, use global grid mode (required for this model)
            TBPTT_latent_state: Latent state from previous TBPTT segment
            TBPTT_prev_true_latent: Previous true latent for delta computation
            input_substeps: Number of sub-steps per forward pass
            compute_tendencies: If False, skip tendency decoder to save memory
        
        Returns:
            In training mode: 8-tuple of tensors
            In inference mode: predicted state tensor
        """
        use_full_ckpt = (
            self.grad_ckpt_level >= 2
            and global_mode
            and self.training
        )

        if use_full_ckpt:
            return self._forward_with_checkpoint(
                x_current_global,
                x_next_global,
                num_rollout_steps=num_rollout_steps,
                decode_freq=decode_freq,
                train=train,
                global_mode=global_mode,
                TBPTT_latent_state=TBPTT_latent_state,
                TBPTT_prev_true_latent=TBPTT_prev_true_latent,
                input_substeps=input_substeps,
                compute_tendencies=compute_tendencies,
            )

        return self._forward_impl(
            x_current_global,
            x_next_global,
            num_rollout_steps=num_rollout_steps,
            decode_freq=decode_freq,
            train=train,
            global_mode=global_mode,
            TBPTT_latent_state=TBPTT_latent_state,
            TBPTT_prev_true_latent=TBPTT_prev_true_latent,
            input_substeps=input_substeps,
            compute_tendencies=compute_tendencies,
        )

    def _forward_with_checkpoint(
        self,
        x_current_global: torch.Tensor,
        x_next_global: Optional[torch.Tensor] = None,
        *,
        num_rollout_steps: Optional[int] = None,
        decode_freq: Optional[int] = None,
        train: Optional[bool] = None,
        global_mode: Optional[bool] = True,
        TBPTT_latent_state: Optional[torch.Tensor] = None,
        TBPTT_prev_true_latent: Optional[torch.Tensor] = None,
        input_substeps: Optional[int] = None,
        compute_tendencies: Optional[bool] = None,
    ):
        """Checkpointed forward pass wrapper."""
        use_next_placeholder = x_next_global is None
        if use_next_placeholder:
            x_next_arg = torch.zeros(1, device=x_current_global.device, dtype=x_current_global.dtype)
        else:
            x_next_arg = x_next_global
        
        # Handle TBPTT tensors for checkpointing
        use_tbptt_latent_placeholder = TBPTT_latent_state is None
        use_tbptt_true_placeholder = TBPTT_prev_true_latent is None
        if use_tbptt_latent_placeholder:
            tbptt_latent_arg = torch.zeros(1, device=x_current_global.device, dtype=x_current_global.dtype)
        else:
            tbptt_latent_arg = TBPTT_latent_state
        if use_tbptt_true_placeholder:
            tbptt_true_arg = torch.zeros(1, device=x_current_global.device, dtype=x_current_global.dtype)
        else:
            tbptt_true_arg = TBPTT_prev_true_latent

        dummy = torch.zeros(1, device=x_current_global.device, requires_grad=True)

        if input_substeps is None:
             input_substeps_arg = torch.tensor(self.input_substeps, device=x_current_global.device)
        else:
             input_substeps_arg = torch.tensor(input_substeps, device=x_current_global.device)

        def body(x_cur: torch.Tensor, x_next: torch.Tensor, tbptt_lat: torch.Tensor, tbptt_true: torch.Tensor, substeps_tensor: torch.Tensor, _dummy: torch.Tensor):
            next_tensor = None if use_next_placeholder else x_next
            tbptt_lat_tensor = None if use_tbptt_latent_placeholder else tbptt_lat
            tbptt_true_tensor = None if use_tbptt_true_placeholder else tbptt_true
            substeps = substeps_tensor.item()
            outputs = self._forward_impl(
                x_cur,
                next_tensor,
                num_rollout_steps=num_rollout_steps,
                decode_freq=decode_freq,
                train=train,
                global_mode=global_mode,
                TBPTT_latent_state=tbptt_lat_tensor,
                TBPTT_prev_true_latent=tbptt_true_tensor,
                input_substeps=substeps,
                compute_tendencies=compute_tendencies,
            )
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            packed, masks = self._pack_checkpoint_outputs(outputs, reference_tensor=x_cur)
            return tuple(packed + masks)

        checkpoint_outputs = checkpoint(body, x_current_global, x_next_arg, tbptt_latent_arg, tbptt_true_arg, input_substeps_arg, dummy, use_reentrant=False)
        restored = self._unpack_checkpoint_outputs(checkpoint_outputs)
        return restored if isinstance(restored, tuple) else tuple(restored)

    def _pack_checkpoint_outputs(self, outputs: tuple, reference_tensor: torch.Tensor):
        packed = []
        masks = []
        for out in outputs:
            if out is None:
                placeholder = torch.zeros(1, device=reference_tensor.device, dtype=reference_tensor.dtype)
                packed.append(placeholder)
                masks.append(torch.zeros(1, device=reference_tensor.device))
            else:
                packed.append(out)
                mask = torch.ones(1, device=out.device)
                masks.append(mask)
        return packed, masks

    def _unpack_checkpoint_outputs(self, checkpoint_outputs: tuple):
        half = len(checkpoint_outputs) // 2
        values = checkpoint_outputs[:half]
        masks = checkpoint_outputs[half:]
        restored = []
        for tensor, mask in zip(values, masks):
            if mask.item() == 0:
                restored.append(None)
            else:
                restored.append(tensor)
        return tuple(restored)

    def _forward_impl(
        self,
        x_current_global: torch.Tensor,
        x_next_global: Optional[torch.Tensor] = None,
        *,
        num_rollout_steps: Optional[int] = None,
        decode_freq: Optional[int] = None,
        train: Optional[bool] = None,
        global_mode: Optional[bool] = True,
        TBPTT_latent_state: Optional[torch.Tensor] = None,
        TBPTT_prev_true_latent: Optional[torch.Tensor] = None,
        input_substeps: Optional[int] = None,
        compute_tendencies: Optional[bool] = None,
    ):
        """Main forward implementation with full training infrastructure."""
        
        # Determine whether to compute tendencies
        if compute_tendencies is None:
            should_compute_tendencies = self.default_compute_tendencies
        else:
            should_compute_tendencies = compute_tendencies
        
        if not global_mode:
            raise ValueError("DISCO_NNorm_GAM2 only supports global_mode=True")
        
        if x_current_global.dim() != 4:
            raise ValueError("x_current_global must have shape (batch, lat, lon, features).")

        batch, height, width, features = x_current_global.shape
        if features != self.M_features:
            raise ValueError(
                f"x_current_global last dimension ({features}) must match initialized M_features ({self.M_features})."
            )
        if height != self.n_lat or width != self.n_lon:
            raise ValueError(
                f"Input spatial dims ({height}, {width}) don't match model ({self.n_lat}, {self.n_lon})"
            )

        if num_rollout_steps is None:
            steps = self.default_num_rollout_steps
        else:
            if num_rollout_steps < 1:
                raise ValueError("num_rollout_steps must be at least 1.")
            steps = num_rollout_steps
            self.default_num_rollout_steps = num_rollout_steps

        if decode_freq is None:
            decode_frequency = self.default_decode_freq
        else:
            if decode_freq < 1:
                raise ValueError("decode_freq must be at least 1.")
            decode_frequency = decode_freq
            self.default_decode_freq = decode_freq

        if train is None:
            training_mode = self.is_training_mode
        else:
            training_mode = bool(train)
            self.is_training_mode = training_mode

        if training_mode:
            if x_next_global is None:
                raise ValueError("x_next_global is required when the model is in training mode.")
            if x_next_global.dim() == 4 and steps == 1:
                x_next_global = x_next_global.unsqueeze(1)
            elif x_next_global.dim() != 5:
                raise ValueError(
                    "x_next_global must have shape (batch, num_rollout_steps, lat, lon, features) in training mode."
                )
            if x_next_global.shape[0] != batch:
                raise ValueError("x_next_global batch dimension must match x_current_global.")
            if x_next_global.shape[1] != steps:
                raise ValueError("x_next_global num_rollout_steps dimension must match the provided value.")
            if x_next_global.shape[2] != height or x_next_global.shape[3] != width:
                raise ValueError("Spatial dimensions of x_next_global must match x_current_global.")
            if x_next_global.shape[4] != self.M_features:
                raise ValueError("x_next_global feature dimension must match initialized M_features.")

        # Use TBPTT latent state if provided, otherwise encode from input
        if TBPTT_latent_state is not None:
            latent_state = TBPTT_latent_state
            encoded_current = None
            initial_true_latent_for_delta = TBPTT_prev_true_latent
        else:
            encoded_current = self._encode_global(x_current_global)
            latent_state = encoded_current
            initial_true_latent_for_delta = encoded_current

        current_substeps = input_substeps if input_substeps is not None else self.input_substeps

        x_next_pred_steps: List[torch.Tensor] = []
        delta_pred_steps: Optional[List[torch.Tensor]] = [] if training_mode else None
        delta_true_steps: Optional[List[torch.Tensor]] = [] if training_mode else None
        reconstructed_true_steps: Optional[List[torch.Tensor]] = [] if training_mode else None
        tend_pred_steps: Optional[List[torch.Tensor]] = [] if training_mode else None
        tend_true_steps: Optional[List[torch.Tensor]] = [] if training_mode else None

        if training_mode and x_next_global is not None:
            # Encode all future steps
            encoded_next_true_list = []
            for step_i in range(steps):
                x_next_step = x_next_global[:, step_i, ...].contiguous()
                with torch.set_grad_enabled(self.teacher_forcing_requires_grad):
                    encoded_step = self._encode_global(x_next_step)
                encoded_next_true_list.append(encoded_step)
            encoded_next_true = torch.stack(encoded_next_true_list, dim=1)

            if not self.teacher_forcing_requires_grad:
                encoded_next_true = encoded_next_true.detach()
        else:
            encoded_next_true = None

        for step_idx in range(steps):
            accumulated_delta_pred = torch.zeros_like(latent_state)
            
            # Sub-stepping loop
            for sub_step in range(current_substeps):
                delta_pred_sub = self._compute_latent_delta(latent_state)
                
                # Update latent state for next sub-step
                learnable_next = latent_state[:, :, :, :-2] + delta_pred_sub[:, :, :, :-2]
                sin_cos = latent_state[:, :, :, -2:]
                latent_state = torch.cat([learnable_next, sin_cos], dim=-1)
                
                # Accumulate total delta for loss computation
                accumulated_delta_pred = accumulated_delta_pred + delta_pred_sub

            delta_pred = accumulated_delta_pred
            encoded_next_pred = latent_state

            # Apply SFNO to full state (uses internal residual connection)
            if self.sfno is not None:
                # SFNO expects (B, C, H, W), latent is (B, H, W, C)
                encoded_permuted = encoded_next_pred.permute(0, 3, 1, 2)
                use_block_ckpt = self.grad_ckpt_level in (1, 3)
                if use_block_ckpt and self.training and encoded_permuted.requires_grad:
                    sfno_out = checkpoint(self.sfno, encoded_permuted, use_reentrant=False)
                else:
                    sfno_out = self.sfno(encoded_permuted)
                encoded_next_pred = sfno_out.permute(0, 2, 3, 1)

            should_decode = (step_idx + 1) % decode_frequency == 0
            
            # Compute delta_true for ALL steps (latent-space loss)
            if (
                training_mode
                and delta_true_steps is not None
                and encoded_next_true is not None
            ):
                encoded_next_true_step = encoded_next_true[:, step_idx, ...].contiguous()
                if not self.teacher_forcing_requires_grad:
                    encoded_next_true_step = encoded_next_true_step.detach()

                if step_idx == 0:
                    prev_true_latent = initial_true_latent_for_delta
                else:
                    prev_true_latent = encoded_next_true[:, step_idx - 1, ...].contiguous()
                if not self.teacher_forcing_requires_grad and prev_true_latent is not None:
                    prev_true_latent = prev_true_latent.detach()

                delta_true = encoded_next_true_step - prev_true_latent
                if not self.teacher_forcing_requires_grad:
                    delta_true = delta_true.detach()
                delta_true_steps.append(delta_true)
            else:
                delta_true = None

            # Always append delta_pred for latent-space loss
            if training_mode and delta_pred_steps is not None:
                delta_pred_steps.append(delta_pred)

            # Decode only at decode_freq intervals (or final step if needed)
            if should_decode or (step_idx == steps - 1 and not x_next_pred_steps):
                decoded_pred = self.decoder(encoded_next_pred)
                
                # Apply spectral filter if enabled
                if self.use_spectral_filter:
                    decoded_pred = self._spectral_filter(decoded_pred)
                
                x_next_pred_steps.append(decoded_pred)
                
                # Only compute decoded outputs at decode steps
                if training_mode:
                    # Predicted tendency (decoded) - skip if compute_tendencies=False
                    if should_compute_tendencies:
                        pred_tend = self.tend_decoder(delta_pred)
                        tend_pred_steps.append(pred_tend)
                    
                    # True tendency and reconstructed_true (decoded)
                    if delta_true is not None and reconstructed_true_steps is not None:
                        if should_compute_tendencies:
                            if self.teacher_forcing_requires_grad:
                                true_tend = self.tend_decoder(delta_true)
                            else:
                                with torch.no_grad():
                                    true_tend = self.tend_decoder(delta_true)
                            tend_true_steps.append(true_tend)
                        
                        # Decode true state at this step
                        encoded_next_true_step = encoded_next_true[:, step_idx, ...].contiguous()
                        with torch.set_grad_enabled(self.teacher_forcing_requires_grad):
                            reconstructed_true_step = self.decoder(encoded_next_true_step)
                        if not self.teacher_forcing_requires_grad:
                            reconstructed_true_step = reconstructed_true_step.detach()
                        reconstructed_true_steps.append(reconstructed_true_step)

        if not x_next_pred_steps:
            decoded_pred = self.decoder(latent_state)
            if self.use_spectral_filter:
                decoded_pred = self._spectral_filter(decoded_pred)
            x_next_pred_steps.append(decoded_pred)

        x_next_pred_rollout = torch.stack(x_next_pred_steps, dim=1)

        if training_mode:
            delta_pred_rollout = torch.stack(delta_pred_steps, dim=1) if delta_pred_steps else None
            if delta_true_steps:
                delta_true_rollout = torch.stack(delta_true_steps, dim=1)
            else:
                delta_true_rollout = None

            if reconstructed_true_steps:
                reconstructed_true_rollout = torch.stack(reconstructed_true_steps, dim=1)
            else:
                reconstructed_true_rollout = None
            if tend_pred_steps:
                tend_pred_rollout = torch.stack(tend_pred_steps, dim=1)
                tend_true_rollout = torch.stack(tend_true_steps, dim=1)
            else:
                tend_pred_rollout = None
                tend_true_rollout = None

            # Get final latent states for TBPTT continuation
            final_pred_latent = latent_state
            final_true_latent = encoded_next_true[:, -1, ...].contiguous() if encoded_next_true is not None else None

            return (
                delta_pred_rollout,
                delta_true_rollout,
                x_next_pred_rollout,
                reconstructed_true_rollout,
                tend_pred_rollout,
                tend_true_rollout,
                final_pred_latent,
                final_true_latent,
            )

        return x_next_pred_rollout
