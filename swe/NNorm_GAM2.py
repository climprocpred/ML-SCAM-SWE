#%%
# run in pytorch3 conda environment
import logging
logging.getLogger("torch._subclasses.fake_tensor").setLevel(logging.CRITICAL)
import os
os.environ['TORCH_LOGS'] = '-fake_tensor'
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import List, Optional
import numpy as np

# Optional SFNO import
try:
    from torch_harmonics.examples.models import SphericalFourierNeuralOperator
    SFNO_AVAILABLE = True
except ImportError:
    SFNO_AVAILABLE = False

# ==============================================================================
# === Core Mixer Components (Combined from both models)                      ===
# ==============================================================================

class HeadProcessing(nn.Module):
    """
    Per-head processing after the shared spatial layer.
    Takes the output from the shared first layer and applies head-specific transformations.
    """
    def __init__(self, shared_hidden: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers = []
        current_dim = shared_hidden
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.GELU())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: (batch_size * head_dim, shared_hidden)
        # Output shape: (batch_size * head_dim, output_dim)
        return self.mlp(x)


class LatModulator(nn.Module):
    """
    Generates gamma (scale) and beta (shift) modulation parameters from sin/cos latitude.
    """
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        # Input is 2 (sin, cos)
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize last layer for identity modulation (gamma=1, beta=0)
        # output_dim is 2 * shared_hidden (first half gamma, second half beta)
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias[:output_dim//2].fill_(1.0) # Gamma
            self.net[-1].bias[output_dim//2:].fill_(0.0) # Beta

    def forward(self, x):
        # x: (B, 2)
        return self.net(x)


class HeadMLP(nn.Module):
    """
    DEPRECATED - Kept for backward compatibility.
    An MLP for a single 'head' in a STANDARD mixer block.
    It mixes spatial information for each channel and returns the updated spatial tokens.
    """
    def __init__(self, num_patches: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        current_dim = num_patches
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.GELU())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, num_patches))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: (batch_size * head_dim, num_patches)
        # Output shape: (batch_size * head_dim, num_patches)
        return self.mlp(x)

class CentralColumnHeadMLP(nn.Module):
    """
    An MLP for a single 'head' in the FINAL mixer block.
    It takes spatial info and computes a single scalar for the central column.
    """
    def __init__(self, num_patches: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        current_dim = num_patches
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.GELU())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, 1)) # Output is a single value
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: (batch_size * head_dim, num_patches)
        # Output shape: (batch_size * head_dim, 1)
        return self.mlp(x)

class StandardMixerBlock(nn.Module):
    """
    A standard mixer block that processes and updates ALL tokens (grid columns) in a patch.
    Uses shared first layer (spatial convolution) with lat input, then per-head processing.
    """
    def __init__(self, model_dim: int, patch_size: int, n_heads: int,
                 head_mlp_hidden_dims: List[int], ffn_hidden_dims: List[int], dropout_rate: float,
                 use_residual: bool = True):
        super().__init__()
        # model_dim is full dimension (e.g., 258), learnable_dim excludes immutable sin/cos lat (256)
        # learnable_dim should be divisible by n_heads for spatial mixing
        self.model_dim = model_dim
        self.learnable_dim = model_dim - 2
        self.head_dim = self.learnable_dim // n_heads  # Spatial mixing on learnable portion only
        self.n_heads = n_heads
        self.num_patches = patch_size * patch_size
        self.use_residual = use_residual
        self.grad_ckpt_level = 0

        # Shared first layer: (num_patches + 2) -> first_hidden
        # Input includes sin/cos lat for latitude-aware spatial mixing
        shared_input_dim = self.num_patches 
        shared_hidden = head_mlp_hidden_dims[0] if head_mlp_hidden_dims else 128
        self.shared_spatial_layer = nn.Linear(shared_input_dim, shared_hidden)

        # Lat modulator: 2 (sin/cos) -> shared_hidden*2 -> 2*shared_hidden (gamma, beta)
        # Using shared_hidden * 2 for hidden dim to allow sufficient capacity
        self.lat_modulator = LatModulator(shared_hidden * 2, shared_hidden * 2)

        # Per-head processing after shared layer
        # Takes shared_hidden -> remaining hidden layers -> num_patches output
        remaining_hidden_dims = head_mlp_hidden_dims[1:] if len(head_mlp_hidden_dims) > 1 else []
        self.head_processing = nn.ModuleList([
            HeadProcessing(shared_hidden, remaining_hidden_dims, self.num_patches)
            for _ in range(n_heads)
        ])

        # FFN takes full model_dim as input, outputs learnable_dim
        ffn_layers = []
        current_dim = model_dim  # Input includes sin/cos lat
        for h_dim in ffn_hidden_dims:
            ffn_layers.append(nn.Linear(current_dim, h_dim))
            ffn_layers.append(nn.GELU())
            current_dim = h_dim
        ffn_layers.append(nn.Linear(current_dim, self.learnable_dim))  # Output excludes sin/cos
        ffn_layers.append(nn.Dropout(dropout_rate))
        self.ffn = nn.Sequential(*ffn_layers)

        # Initialize the weights for the last 2 channels (sin/cos) of the first FFN layer to zero
        # This helps stability by starting with latitude-agnostic channel mixing
        with torch.no_grad():
            self.ffn[0].weight[:, -2:] = 0.0

    def set_grad_ckpt_level(self, level: int) -> None:
        self.grad_ckpt_level = level

    def forward(self, x):
        # Input x shape: (batch_of_patches, num_patches, model_dim)
        # batch_of_patches = B_orig * H * W (each is a patch from a different grid cell)
        # Last 2 dims are immutable sin/cos latitude
        B = x.shape[0]  # batch_of_patches
        
        # Extract sin/cos from CENTRAL column - this is the lat of the grid cell we're computing for
        central_idx = self.num_patches // 2
        sin_cos_central = x[:, central_idx, -2:]  # (B, 2) - central column's sin/cos
        
        # Get all sin/cos for output (we'll re-append these unchanged)
        sin_cos_all = x[:, :, -2:]  # (B, num_patches, 2)
        
        # Learnable portion for spatial mixing
        x_learn = x[:, :, :-2]  # (B, num_patches, learnable_dim)
        x_res = x_learn

        # Split learnable portion into heads
        x_split = torch.split(x_learn, self.head_dim, dim=-1)  # n_heads tuples of (B, num_patches, head_dim)
        
        head_outputs = []
        for i in range(self.n_heads):
            # Input: (B, num_patches, head_dim)
            # Permute to (B, head_dim, num_patches) to operate on the spatial dim (last dim)
            head_chunk = x_split[i].permute(0, 2, 1)
            
            # Shared spatial layer: (B, head_dim, 9) -> (B, head_dim, shared_hidden)
            head_input = head_chunk
            
            # Shared spatial layer
            if (
                self.grad_ckpt_level >= 1
                and self.training
                and head_input.requires_grad
            ):
                shared_out = checkpoint(self.shared_spatial_layer, head_input, use_reentrant=False)
            else:
                shared_out = self.shared_spatial_layer(head_input)
            
            # Apply Latitude Modulation (FiLM)
            # 1. Compute gamma/beta from central sin/cos: (B, 2) -> (B, 2*shared_hidden)
            mod_params = self.lat_modulator(sin_cos_central) 
            # 2. Split into gamma/beta: (B, shared_hidden) each
            gamma, beta = torch.chunk(mod_params, 2, dim=-1)
            # 3. Use broadcasting: (B, head_dim, shared_hidden) * (B, 1, shared_hidden)
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            
            # 4. Modulate
            shared_out = shared_out * gamma + beta

            # Per-head processing: (B, head_dim, shared_hidden) -> (B, head_dim, num_patches)
            if (
                self.grad_ckpt_level >= 1
                and self.training
                and shared_out.requires_grad
            ):
                processed = checkpoint(self.head_processing[i], shared_out, use_reentrant=False)
            else:
                processed = self.head_processing[i](shared_out)

            # Permute back: (B, head_dim, num_patches) -> (B, num_patches, head_dim)
            processed_chunk = processed.permute(0, 2, 1)
            head_outputs.append(processed_chunk)

        x_learn = torch.cat(head_outputs, dim=-1)  # (B, num_patches, learnable_dim)
        if self.use_residual:
            x_learn = x_learn + x_res  # First residual connection (on learnable portion)

        # Reconstruct full tensor for FFN input (includes sin/cos as context)
        x_full = torch.cat([x_learn, sin_cos_all], dim=-1)  # (B, num_patches, model_dim)

        x_res = x_learn  # Save learnable portion for residual
        x_learn = self.ffn(x_full)  # FFN: model_dim -> learnable_dim
        x_learn = x_learn + x_res  # Second residual connection (on learnable portion)
        
        # Re-append immutable sin/cos for output
        return torch.cat([x_learn, sin_cos_all], dim=-1)  # (B, num_patches, model_dim)

class FinalMixerBlock(nn.Module):
    """
    A specialized mixer block that only computes the feature for the central column.
    Uses shared first layer (spatial convolution) with lat input, then per-head processing.
    """
    def __init__(self, model_dim: int, patch_size: int, n_heads: int,
                 head_mlp_hidden_dims: List[int], ffn_hidden_dims: List[int], dropout_rate: float):
        super().__init__()
        # model_dim is full dimension (e.g., 258), learnable_dim excludes immutable sin/cos lat (256)
        # learnable_dim should be divisible by n_heads for spatial mixing
        self.model_dim = model_dim
        self.learnable_dim = model_dim - 2
        self.head_dim = self.learnable_dim // n_heads  # Spatial mixing on learnable portion only
        self.n_heads = n_heads
        self.num_patches = patch_size * patch_size
        self.grad_ckpt_level = 0

        # Shared first layer: (num_patches + 2) -> first_hidden
        # Input includes sin/cos lat for latitude-aware spatial mixing
        shared_input_dim = self.num_patches 
        shared_hidden = head_mlp_hidden_dims[0] if head_mlp_hidden_dims else 128
        self.shared_spatial_layer = nn.Linear(shared_input_dim, shared_hidden)

        # Lat modulator: 2 (sin/cos) -> shared_hidden*2 -> 2*shared_hidden (gamma, beta)
        # Using shared_hidden * 2 for hidden dim to allow sufficient capacity
        self.lat_modulator = LatModulator(shared_hidden * 2, shared_hidden * 2)
        
        # Per-head processing after shared layer - outputs 1 value (central column only)
        remaining_hidden_dims = head_mlp_hidden_dims[1:] if len(head_mlp_hidden_dims) > 1 else []
        self.head_processing = nn.ModuleList([
            HeadProcessing(shared_hidden, remaining_hidden_dims, 1)  # Output is 1 (scalar)
            for _ in range(n_heads)
        ])

        # FFN takes full model_dim as input, outputs learnable_dim
        ffn_layers = []
        current_dim = model_dim  # Input includes sin/cos lat
        for h_dim in ffn_hidden_dims:
            ffn_layers.append(nn.Linear(current_dim, h_dim))
            ffn_layers.append(nn.GELU())
            current_dim = h_dim
        ffn_layers.append(nn.Linear(current_dim, self.learnable_dim))  # Output excludes sin/cos
        ffn_layers.append(nn.Dropout(dropout_rate))
        self.ffn = nn.Sequential(*ffn_layers)

        # Initialize the weights for the last 2 channels (sin/cos) of the first FFN layer to zero
        with torch.no_grad():
            self.ffn[0].weight[:, -2:] = 0.0

    def set_grad_ckpt_level(self, level: int) -> None:
        self.grad_ckpt_level = level

    def forward(self, x):
        # Input x shape: (batch_of_patches, num_patches, model_dim)
        # Last 2 dims are immutable sin/cos latitude
        central_idx = self.num_patches // 2
        
        # Extract immutable sin/cos for the central column (original values)
        sin_cos_central = x[:, central_idx, -2:]  # (B, 2) - saved for re-appending
        
        # Learnable portion for all patches
        x_learn = x[:, :, :-2]  # (B, num_patches, learnable_dim)
        
        # Residual is the central column's learnable portion
        x_res = x_learn[:, central_idx, :]  # (B, learnable_dim)

        # Split learnable portion into heads
        x_split = torch.split(x_learn, self.head_dim, dim=-1)

        head_outputs = []
        for i in range(self.n_heads):
            head_chunk = x_split[i].permute(0, 2, 1)  # -> (B, head_dim, num_patches)
            
            # Shared spatial layer: (B, head_dim, num_patches) -> (B, head_dim, shared_hidden)
            head_input = head_chunk
            
            # Shared spatial layer
            if (
                self.grad_ckpt_level >= 1
                and self.training
                and head_input.requires_grad
            ):
                shared_out = checkpoint(self.shared_spatial_layer, head_input, use_reentrant=False)
            else:
                shared_out = self.shared_spatial_layer(head_input)
            
            # Apply Latitude Modulation (FiLM)
            # 1. Compute gamma/beta from central sin/cos: (B, 2) -> (B, 2*shared_hidden)
            mod_params = self.lat_modulator(sin_cos_central) 
            # 2. Split into gamma/beta: (B, shared_hidden) each
            gamma, beta = torch.chunk(mod_params, 2, dim=-1)
            # 3. Use broadcasting: (B, head_dim, shared_hidden) * (B, 1, shared_hidden)
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            
            # 4. Modulate
            shared_out = shared_out * gamma + beta
            
            # Per-head processing: (B, head_dim, shared_hidden) -> (B, head_dim, 1)
            if (
                self.grad_ckpt_level >= 1
                and self.training
                and shared_out.requires_grad
            ):
                center_val = checkpoint(self.head_processing[i], shared_out, use_reentrant=False)
            else:
                center_val = self.head_processing[i](shared_out)  # -> (B, head_dim, 1)

            # Output (B, head_dim)
            head_outputs.append(center_val.squeeze(-1))

        x_mixed = torch.cat(head_outputs, dim=-1)  # (B, learnable_dim)
        x_learn = x_mixed + x_res  # First residual connection (on learnable portion)

        # Reconstruct full tensor for FFN input (includes sin/cos as context)
        x_full = torch.cat([x_learn, sin_cos_central], dim=-1)  # (B, model_dim)

        x_res = x_learn  # Save learnable portion for residual
        x_learn = self.ffn(x_full)  # FFN: model_dim -> learnable_dim
        x_learn = x_learn + x_res  # Second residual connection (on learnable portion)
        
        # Re-append the ORIGINAL immutable sin/cos from central column
        return torch.cat([x_learn, sin_cos_central], dim=-1)  # (B, model_dim)

class SpectralFilter(nn.Module):
    """Learnable radially-symmetric spectral filter.
    
    Applies 2D FFT (with latitude mirroring for proper boundary conditions),
    multiplies by learnable weights that depend only on wavenumber magnitude |k|,
    and applies inverse FFT.
    Initialized to identity (weights=1.0) for fine-tuning existing models.
    """
    def __init__(self, n_lat: int, n_lon: int, init_value: float = 1.0):
        super().__init__()
        # After lat mirroring, spectral shape is (2*n_lat, n_lon//2+1) for rfft2
        n_lat_spectral = 2 * n_lat
        n_lon_spectral = n_lon // 2 + 1
        
        # Compute wavenumber frequencies (normalized to [0, 0.5])
        ky = torch.fft.fftfreq(n_lat_spectral)  # [-0.5, 0.5)
        kx = torch.fft.rfftfreq(n_lon)          # [0, 0.5]
        KY, KX = torch.meshgrid(ky, kx, indexing='ij')
        K_mag = torch.sqrt(KY**2 + KX**2)  # (n_lat_spectral, n_lon_spectral)
        
        # Discretize k magnitude into bins
        # Max possible k magnitude is sqrt(0.5^2 + 0.5^2) ≈ 0.707
        # Use enough bins to capture the full range with good resolution
        num_k_bins = max(n_lat_spectral, n_lon_spectral) // 2 + 1
        self.num_k_bins = num_k_bins
        
        # Map each spectral position to a k bin index
        k_bin_indices = (K_mag / 0.5 * (num_k_bins - 1)).round().long().clamp(0, num_k_bins - 1)
        self.register_buffer('k_bin_indices', k_bin_indices)
        
        # Learnable 1D weights for each k bin, initialized to identity
        self.k_weights = nn.Parameter(torch.full((num_k_bins,), init_value))
    
    def forward(self, x):
        # x shape: (B, H, W, C) where H=lat, W=lon
        original_dtype = x.dtype
        
        # 1. Mirror latitude to eliminate pole discontinuity
        x_mirrored = torch.cat([x, x.flip(dims=[1])], dim=1)  # (B, 2H, W, C)
        
        # 2. Permute to (B, C, 2H, W) for FFT
        x_perm = x_mirrored.permute(0, 3, 1, 2)
        
        # 3. Cast to float32 for FFT (doesn't support BFloat16)
        x_perm = x_perm.float()
        
        # 4. Apply 2D real FFT
        x_fft = torch.fft.rfft2(x_perm, norm='ortho')  # (B, C, 2H, W//2+1)
        
        # 5. Build 2D filter weights from 1D k_weights using precomputed indices
        filter_2d = self.k_weights[self.k_bin_indices]  # (n_lat_spectral, n_lon_spectral)
        
        # 6. Multiply by filter weights (broadcast over B, C)
        x_fft = x_fft * filter_2d.unsqueeze(0).unsqueeze(0)
        
        # 7. Inverse FFT
        x_filtered = torch.fft.irfft2(x_fft, s=x_perm.shape[-2:], norm='ortho')
        
        # 8. Cast back to original dtype
        x_filtered = x_filtered.to(original_dtype)
        
        # 9. Permute back to (B, 2H, W, C)
        x_filtered = x_filtered.permute(0, 2, 3, 1)
        
        # 10. Take only the first half (remove mirrored portion) and make contiguous
        return x_filtered[:, :x.shape[1], :, :].contiguous()


# ==============================================================================
# === Encoder / Decoder Components                                         ===
# ==============================================================================

class Encoder(nn.Module):
    """Encodes the physical state into a latent space at each grid point.
    
    Outputs model_dim - 2 dimensions (the learnable portion).
    Sin/cos latitude features are appended separately by _encode_global.
    """
    def __init__(self, M_features, model_dim, encoder_hidden_dims: List[int]):
        super().__init__()
        self.grad_ckpt_level = 0
        # Output dimension is model_dim - 2 (excludes sin/cos lat which is appended after)
        learnable_dim = model_dim - 2
        layers = []
        current_dim = M_features
        for h_dim in encoder_hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.GELU())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, learnable_dim))
        self.encoder_mlp = nn.Sequential(*layers)

    def set_grad_ckpt_level(self, level: int) -> None:
        self.grad_ckpt_level = level

    def forward(self, x):
        # Input can be (B, H, W, M) or (B*H*W, M)
        use_ckpt = self.grad_ckpt_level >= 3 and self.training and x.requires_grad
        if x.dim() == 4:
            B, H, W, M = x.shape
            x_flat = x.view(-1, M)
            if use_ckpt:
                encoded_flat = checkpoint(self.encoder_mlp, x_flat, use_reentrant=False)
            else:
                encoded_flat = self.encoder_mlp(x_flat)
            return encoded_flat.view(B, H, W, -1)
        elif x.dim() == 2:
            if use_ckpt:
                return checkpoint(self.encoder_mlp, x, use_reentrant=False)
            else:
                return self.encoder_mlp(x)
        else:
            raise ValueError(f"Unsupported input dimension for Encoder: {x.dim()}")

class Decoder(nn.Module):
    """Decodes the latent state back to the physical state at each grid point."""
    def __init__(self, model_dim, M_features, decoder_hidden_dims: List[int]):
        super().__init__()
        self.grad_ckpt_level = 0
        layers = []
        current_dim = model_dim
        for h_dim in decoder_hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.GELU())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, M_features))
        self.decoder_mlp = nn.Sequential(*layers)

    def set_grad_ckpt_level(self, level: int) -> None:
        self.grad_ckpt_level = level

    def forward(self, z):
        # Input can be (B, H, W, D) or (B*H*W, D)
        use_ckpt = self.grad_ckpt_level >= 3 and self.training and z.requires_grad
        if z.dim() > 2:
             B, H, W, D = z.shape
             z_flat = z.reshape(-1, D)  # Use reshape instead of view for non-contiguous tensors
             if use_ckpt:
                 decoded_flat = checkpoint(self.decoder_mlp, z_flat, use_reentrant=False)
             else:
                 decoded_flat = self.decoder_mlp(z_flat)
             return decoded_flat.reshape(B, H, W, -1)
        elif z.dim() == 2:
             if use_ckpt:
                 return checkpoint(self.decoder_mlp, z, use_reentrant=False)
             else:
                 return self.decoder_mlp(z)
        else:
             raise ValueError(f"Unsupported input dimension for Decoder: {z.dim()}")


# ==============================================================================
# === Main Global Latent Model (Refactored)                                  ===
# ==============================================================================

class NNorm_LatentGlobalAtmosMixer2(nn.Module):
    """
    A global atmospheric model that learns tendencies in a latent space.

    It encodes a global grid into a latent representation, then applies a
    sequence of mixer blocks to predict the latent tendency for every grid cell.
    The final latent state is decoded back to the physical state.

    """
    def __init__(self, M_features: int, model_dim: int, n_heads: int,
                 num_mixer_blocks: int, head_mlp_hidden_dims: List[int],
                 ffn_hidden_dims: List[int], encoder_hidden_dims: List[int],
                 dropout_rate: float = 0.1, grad_ckpt_level: int = 1,
                 num_correction_blocks: int = 0,
                 correction_head_mlp_hidden_dims: Optional[List[int]] = None,
                 correction_ffn_hidden_dims: Optional[List[int]] = None,
                 use_spectral_filter: bool = False,
                 n_lat: int = 192,
                 n_lon: int = 288,
                 use_sfno: bool = False,
                 sfno_embed_dim: int = 16,
                 sfno_num_layers: int = 1,
                 sfno_use_mlp: bool = False,
                 sfno_scale_factor: int = 2,
                 sfno_mlp_ratio: int = 4,
                 patch_size: int = 3,
                 use_mixer_encoder: bool = False,
                 encoder_mixer_n_heads: Optional[int] = None,
                 encoder_mixer_head_dims: Optional[List[int]] = None,
                 encoder_mixer_ffn_dims: Optional[List[int]] = None,
                 num_encoder_mixer_blocks: int = 1,
                 input_substeps: int = 1,
                 **kwargs):
        super().__init__()
        self.M_features = M_features
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.pad_size = patch_size // 2
        self.default_num_rollout_steps = 1
        self.default_decode_freq = 1
        self.is_training_mode = True
        self.teacher_forcing_requires_grad = kwargs.pop('teacher_forcing_requires_grad', True)
        self.grad_ckpt_level = grad_ckpt_level
        if kwargs:
            unexpected = ", ".join(kwargs.keys())
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        if self.grad_ckpt_level not in (0, 1, 2, 3):
            raise ValueError("grad_ckpt_level must be one of {0, 1, 2, 3}.")

        if num_mixer_blocks < 1:
            raise ValueError("num_mixer_blocks must be at least 1.")

        self.input_substeps = input_substeps

        # --- 1. Encoder and Decoder ---
        self.use_mixer_encoder = use_mixer_encoder
        if use_mixer_encoder:
            # Inlined MixerEncoder: input projection + mixer blocks
            # Use main config dims as defaults if not specified
            enc_n_heads = encoder_mixer_n_heads if encoder_mixer_n_heads is not None else n_heads
            enc_head_dims = encoder_mixer_head_dims if encoder_mixer_head_dims is not None else head_mlp_hidden_dims
            enc_ffn_dims = encoder_mixer_ffn_dims if encoder_mixer_ffn_dims is not None else ffn_hidden_dims
            
            # Project M_features -> learnable_dim (model_dim - 2) before spatial mixing
            # Sin/cos lat will be appended from the input in _encode_global
            self.encoder_input_proj = nn.Linear(M_features, model_dim - 2)
            
            # Build encoder mixer blocks (same pattern as main mixer)
            encoder_blocks_list = []
            for block_idx in range(num_encoder_mixer_blocks - 1):
                encoder_blocks_list.append(
                    StandardMixerBlock(
                        model_dim, patch_size, enc_n_heads, enc_head_dims, enc_ffn_dims,
                        dropout_rate, use_residual=(block_idx != 0)
                    )
                )
            encoder_blocks_list.append(
                FinalMixerBlock(model_dim, patch_size, enc_n_heads, enc_head_dims, enc_ffn_dims, dropout_rate)
            )
            self.encoder_mixer_blocks = nn.ModuleList(encoder_blocks_list)
            self.encoder = None  # Not used when mixer encoder is enabled
        else:
            self.encoder_input_proj = None
            self.encoder_mixer_blocks = None
            self.encoder = Encoder(M_features, model_dim, encoder_hidden_dims)
        self.decoder = Decoder(model_dim, M_features, encoder_hidden_dims[::-1]) # Symmetrical decoder
        self.tend_decoder = Decoder(model_dim, M_features, encoder_hidden_dims[::-1]) # Symmetrical decoder


        # --- 2. Mixer Blocks ---
        mixer_blocks_list = []
        # Add N-1 Standard Mixer Blocks
        for block_idx in range(num_mixer_blocks - 1):
            mixer_blocks_list.append(
                StandardMixerBlock(
                    model_dim,
                    self.patch_size,
                    n_heads,
                    head_mlp_hidden_dims,
                    ffn_hidden_dims,
                    dropout_rate,
                    use_residual=(block_idx != 0)
                )
            )
        # Add the Final Mixer Block to collapse the spatial info
        mixer_blocks_list.append(
            FinalMixerBlock(model_dim, self.patch_size, n_heads, head_mlp_hidden_dims, ffn_hidden_dims, dropout_rate)
        )
        self.mixer = nn.Sequential(*mixer_blocks_list)

        # --- 3. Correction Layers ---
        self.num_correction_blocks = num_correction_blocks
        if self.num_correction_blocks > 0:
            # Use provided dims or default to main mixer dims
            c_head_dims = correction_head_mlp_hidden_dims if correction_head_mlp_hidden_dims is not None else head_mlp_hidden_dims
            c_ffn_dims = correction_ffn_hidden_dims if correction_ffn_hidden_dims is not None else ffn_hidden_dims
            
            correction_blocks_list = []
            # Add N-1 Standard Mixer Blocks
            for block_idx in range(num_correction_blocks - 1):
                correction_blocks_list.append(
                    StandardMixerBlock(
                        model_dim,
                        self.patch_size,
                        n_heads,
                        c_head_dims,
                        c_ffn_dims,
                        dropout_rate,
                        use_residual=True # Always use residual for correction layers? Or follow same pattern? 
                                          # Let's assume yes, as they are refining an existing signal.
                    )
                )
            # Add Final Mixer Block
            correction_blocks_list.append(
                FinalMixerBlock(model_dim, self.patch_size, n_heads, c_head_dims, c_ffn_dims, dropout_rate)
            )
            self.correction_layers = nn.Sequential(*correction_blocks_list)
        else:
            self.correction_layers = None

        # --- 4. Spectral Filter (Optional) ---
        self.use_spectral_filter = use_spectral_filter
        if self.use_spectral_filter:
            self.spectral_filter = SpectralFilter(n_lat=n_lat, n_lon=n_lon)
        else:
            self.spectral_filter = None

        # --- 5. SFNO Post-Processor (Optional) ---
        self.use_sfno = use_sfno
        if self.use_sfno:
            if not SFNO_AVAILABLE:
                raise ImportError(
                    "torch_harmonics is required for SFNO. Install with: pip install torch_harmonics"
                )
            self.sfno = SphericalFourierNeuralOperator(
                img_size=(n_lat, n_lon),
                in_chans=model_dim,
                out_chans=model_dim,
                embed_dim=sfno_embed_dim,
                num_layers=sfno_num_layers,
                use_mlp=sfno_use_mlp,
                mlp_ratio=sfno_mlp_ratio,
                residual_prediction=True,  # SFNO handles skip connection internally
                scale_factor=sfno_scale_factor,
                grid="equiangular",
                normalization_layer="none",
                pos_embed="learnable lat"
            )
        else:
            self.sfno = None

        self._set_mixer_grad_ckpt_level(self.grad_ckpt_level)
        # Initialize encoder/decoder checkpoint levels
        if self.use_mixer_encoder and self.encoder_mixer_blocks is not None:
            block_level = 1 if self.grad_ckpt_level in (1, 3) else 0
            for block in self.encoder_mixer_blocks:
                if hasattr(block, "set_grad_ckpt_level"):
                    block.set_grad_ckpt_level(block_level)
        elif self.encoder is not None:
            self.encoder.set_grad_ckpt_level(self.grad_ckpt_level)
        self.decoder.set_grad_ckpt_level(self.grad_ckpt_level)
        self.tend_decoder.set_grad_ckpt_level(self.grad_ckpt_level)

    def set_teacher_forcing_grad(self, enabled: bool) -> None:
        self.teacher_forcing_requires_grad = bool(enabled)

    def set_grad_ckpt_level(self, level: int) -> None:
        if level not in (0, 1, 2, 3):
            raise ValueError("grad_ckpt_level must be one of {0, 1, 2, 3}.")
        self.grad_ckpt_level = level
        self._set_mixer_grad_ckpt_level(level)
        # Propagate to encoder/decoder for level 3
        if self.use_mixer_encoder and self.encoder_mixer_blocks is not None:
            # Set grad ckpt level for encoder mixer blocks
            block_level = 1 if level in (1, 3) else 0
            for block in self.encoder_mixer_blocks:
                if hasattr(block, "set_grad_ckpt_level"):
                    block.set_grad_ckpt_level(block_level)
        elif self.encoder is not None:
            self.encoder.set_grad_ckpt_level(level)
        self.decoder.set_grad_ckpt_level(level)
        self.tend_decoder.set_grad_ckpt_level(level)

    def _set_mixer_grad_ckpt_level(self, level: int) -> None:
        block_level = 1 if level in (1, 3) else 0
        for block in self.mixer:
            if hasattr(block, "set_grad_ckpt_level"):
                block.set_grad_ckpt_level(block_level)
        
        if self.correction_layers is not None:
            for block in self.correction_layers:
                if hasattr(block, "set_grad_ckpt_level"):
                    block.set_grad_ckpt_level(block_level)

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
    ):
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
    ):
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
            )
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            packed, masks = self._pack_checkpoint_outputs(outputs, reference_tensor=x_cur)
            return tuple(packed + masks)

        checkpoint_outputs = checkpoint(body, x_current_global, x_next_arg, tbptt_latent_arg, tbptt_true_arg, input_substeps_arg, dummy, use_reentrant=False)
        restored = self._unpack_checkpoint_outputs(checkpoint_outputs)
        return restored if isinstance(restored, tuple) else tuple(restored)

    def _pack_checkpoint_outputs(self, outputs: tuple[torch.Tensor, ...], reference_tensor: torch.Tensor):
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

    def _unpack_checkpoint_outputs(self, checkpoint_outputs: tuple[torch.Tensor, ...]):
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


    def _apply_custom_padding(self, x):
        # Input x shape: (B, H, W, C)
        # Use circular padding for longitude (dim 2 of CHW format)
        # Pad by self.pad_size on each side
        x_circ = F.pad(x.permute(0, 3, 1, 2), (self.pad_size, self.pad_size, 0, 0), mode='circular').permute(0, 2, 3, 1)
        
        # For polar padding: flip latitude order within the padding rows, flip longitude, and roll by 180°
        # This correctly handles the geometry when crossing the poles
        W = x_circ.shape[2]
        half_W = W // 2
        
        # North pole: take first pad_size rows, flip both lat and lon order, roll by 180°
        north_pole_rows = x_circ[:, :self.pad_size, :, :]  # (B, pad_size, W, C)
        north_pole_rows = torch.flip(north_pole_rows, dims=[1, 2])  # Flip lat order and lon
        north_pole_rows = torch.roll(north_pole_rows, shifts=half_W, dims=2)  # 180° shift
        
        # South pole: take last pad_size rows, flip both lat and lon order, roll by 180°
        south_pole_rows = x_circ[:, -self.pad_size:, :, :]  # (B, pad_size, W, C)
        south_pole_rows = torch.flip(south_pole_rows, dims=[1, 2])  # Flip lat order and lon
        south_pole_rows = torch.roll(south_pole_rows, shifts=half_W, dims=2)  # 180° shift
        
        x_padded = torch.cat([north_pole_rows, x_circ, south_pole_rows], dim=1)
        return x_padded

    def _extract_patches(self, grid_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extracts patches from a grid tensor (B, H, W, C).
        Patch size is determined by self.patch_size (e.g., 3x3 or 5x5).
        Returns patches flattened for mixer input: (B*H*W, patch_size^2, C).
        """
        if grid_tensor.dim() != 4:
            raise ValueError("grid_tensor must have shape (batch, lat, lon, features).")

        padded = self._apply_custom_padding(grid_tensor)
        padded_permuted = padded.permute(0, 3, 1, 2)

        num_patches = self.patch_size * self.patch_size
        feature_dim = grid_tensor.shape[-1]  # Get actual feature dimension from input
        patches = padded_permuted.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)
        # patches shape: (B, C, H, W, patch_size, patch_size)
        # We need (B*H*W, num_patches, C) for the mixer
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, feature_dim, num_patches)
        patches = patches.permute(0, 2, 1)  # -> (B*H*W, num_patches, C)
        return patches

    def _encode_global(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a global grid using either mixer encoder or standard MLP Encoder.
        
        Args:
            x: (B, H, W, M_features) - physical state grid
               Last 2 features are sin/cos of latitude (immutable)
        
        Returns:
            encoded: (B, H, W, model_dim) - latent state grid
                     Last 2 dims are sin/cos lat (immutable, copied from input)
        """
        B, H, W, _ = x.shape
        
        # Extract immutable sin/cos latitude from input (last 2 features)
        sin_cos_lat = x[:, :, :, -2:]  # (B, H, W, 2)
        
        if self.use_mixer_encoder and self.encoder_mixer_blocks is not None:
            # Extract patches for each grid cell: (B*H*W, num_patches, M_features)
            patches = self._extract_patches(x)
            
            # Extract sin/cos for each patch's center (for appending after projection)
            # The center of each patch contains the sin/cos for that grid cell
            central_idx = (self.patch_size * self.patch_size - 1) // 2
            sin_cos_patches = patches[:, central_idx, -2:]  # (B*H*W, 2)
            
            # Project to learnable_dim (254d)
            use_ckpt = self.grad_ckpt_level >= 3 and self.training and patches.requires_grad
            if use_ckpt:
                x_enc = checkpoint(self.encoder_input_proj, patches, use_reentrant=False)
            else:
                x_enc = self.encoder_input_proj(patches)  # (B*H*W, num_patches, 254)
            
            # Append sin/cos to each patch position for mixer input
            # Need to broadcast sin_cos_patches to all patch positions
            num_patches = self.patch_size * self.patch_size
            sin_cos_expanded = sin_cos_patches.unsqueeze(1).expand(-1, num_patches, -1)  # (B*H*W, num_patches, 2)
            x_enc = torch.cat([x_enc, sin_cos_expanded], dim=-1)  # (B*H*W, num_patches, 256)
            
            # Apply encoder mixer blocks
            use_block_ckpt = self.grad_ckpt_level in (1, 3)
            for block in self.encoder_mixer_blocks:
                if use_block_ckpt and self.training and x_enc.requires_grad:
                    x_enc = checkpoint(block, x_enc, use_reentrant=False)
                else:
                    x_enc = block(x_enc)
            
            # Reshape back to grid: (B, H, W, model_dim)
            return x_enc.view(B, H, W, self.model_dim)
        else:
            # Standard pointwise encoding (outputs 254d)
            encoded_learnable = self.encoder(x)  # (B, H, W, 254)
            # Append sin/cos lat to get full latent (256d)
            return torch.cat([encoded_learnable, sin_cos_lat], dim=-1)

    def _compute_latent_delta(self, latent_grid: torch.Tensor) -> torch.Tensor:
        if latent_grid.dim() != 4:
            raise ValueError("latent_grid must have shape (batch, lat, lon, model_dim).")

        batch, height, width, _ = latent_grid.shape

        # --- 1. Main Mixer Prediction ---
        patches = self._extract_patches(latent_grid)

        # Apply gradient checkpointing to each mixer block individually
        delta_flat = patches
        use_block_ckpt = self.grad_ckpt_level in (1, 3)
        for mixer_block in self.mixer:
            if use_block_ckpt and self.training and delta_flat.requires_grad:
                delta_flat = checkpoint(mixer_block, delta_flat, use_reentrant=False)
            else:
                delta_flat = mixer_block(delta_flat)
        
        # delta_flat is now (B*H*W, C)
        
        # --- 2. Correction Layers (Optional) ---
        if self.correction_layers is not None:
            # Reshape delta_flat back to grid to extract patches
            delta_grid = delta_flat.view(batch, height, width, self.model_dim)
            
            # Extract patches from the PREDICTED TENDENCY
            patches_delta = self._extract_patches(delta_grid)
            
            delta_corr_flat = patches_delta
            for corr_block in self.correction_layers:
                if use_block_ckpt and self.training and delta_corr_flat.requires_grad:
                    delta_corr_flat = checkpoint(corr_block, delta_corr_flat, use_reentrant=False)
                else:
                    delta_corr_flat = corr_block(delta_corr_flat)
            
            # Add correction to the initial prediction
            # The correction layers are residual blocks, so they already contain the input signal.
            # We just update delta_flat with the refined output.
            delta_flat = delta_corr_flat

        # --- 3. Spectral Filtering (Optional) ---
        if self.spectral_filter is not None:
            delta_grid = delta_flat.view(batch, height, width, self.model_dim)
            if use_block_ckpt and self.training and delta_grid.requires_grad:
                delta_grid = checkpoint(self.spectral_filter, delta_grid, use_reentrant=False)
            else:
                delta_grid = self.spectral_filter(delta_grid)
            delta_flat = delta_grid.view(-1, self.model_dim)

        # Reshape to grid
        delta_out = delta_flat.view(batch, height, width, self.model_dim)
        
        # CRITICAL: The last 2 channels are immutable sin/cos latitude.
        # The mixer blocks pass them through, so 'delta' currently contains the full value.
        # We must zero out their tendency to prevent exponential drift during rollout (lat = lat + lat).
        # We create a new tensor to avoid modifying the output of the mixer in-place (which might be used elsewhere).
        delta_final = delta_out.clone()
        delta_final[..., -2:] = 0.0
        
        return delta_final

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
    ):
        # if in global mode, do forward pass wiht infrasturcture for global prediction and rollout
        if global_mode:
            if x_current_global.dim() != 4:
                raise ValueError("x_current_global must have shape (batch, lat, lon, features).")

            batch, height, width, features = x_current_global.shape
            if features != self.M_features:
                raise ValueError(
                    f"x_current_global last dimension ({features}) must match initialized M_features ({self.M_features})."
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
                encoded_current = None  # Not used when continuing from TBPTT state
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
                # Encode all future steps - loop to use _encode_global which handles MixerEncoder
                encoded_next_true_list = []
                for step_i in range(steps):
                    x_next_step = x_next_global[:, step_i, ...]  # (B, H, W, M_features)
                    with torch.set_grad_enabled(self.teacher_forcing_requires_grad):
                        encoded_step = self._encode_global(x_next_step)
                    encoded_next_true_list.append(encoded_step)
                encoded_next_true = torch.stack(encoded_next_true_list, dim=1)  # (B, steps, H, W, model_dim)
                # NOTE: reconstructed_true is now computed per-step inside the loop at decode_freq intervals

                if not self.teacher_forcing_requires_grad:
                    encoded_next_true = encoded_next_true.detach()
            else:
                encoded_next_true = None

            for step_idx in range(steps):
                accumulated_delta_pred = torch.zeros_like(latent_state)
                # Sub-stepping loop
                for sub_step in range(current_substeps):
                     delta_pred_sub = self._compute_latent_delta(latent_state)
                     
                     # Update latent state for next sub-step (or final step)
                     # Delta only applies to learnable portion (first 254d), sin/cos is preserved
                     learnable_next = latent_state[:, :, :, :-2] + delta_pred_sub[:, :, :, :-2]
                     sin_cos = latent_state[:, :, :, -2:]
                     latent_state = torch.cat([learnable_next, sin_cos], dim=-1)
                     
                     # Accumulate total delta for loss computation
                     accumulated_delta_pred = accumulated_delta_pred + delta_pred_sub

                # Use accumulated delta for auxiliary loss and checks (replaces single-step delta_pred)
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

                    # print('SFNO USED')

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
                    x_next_pred_steps.append(decoded_pred)
                    
                    # Only compute decoded outputs at decode steps
                    if training_mode:
                        # Predicted tendency (decoded)
                        pred_tend = self.tend_decoder(delta_pred)
                        tend_pred_steps.append(pred_tend)
                        
                        # True tendency and reconstructed_true (decoded)
                        if delta_true is not None and reconstructed_true_steps is not None:
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

                            if not self.teacher_forcing_requires_grad:
                                reconstructed_true_step = reconstructed_true_step.detach()
                            reconstructed_true_steps.append(reconstructed_true_step)

                # latent_state is already updated in the sub-step loop

            if not x_next_pred_steps:
                decoded_pred = self.decoder(latent_state)
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

        # if not in globa mode, use single column training flow.
        # x_current_global and x_next_global will be given as (B, 3, 3, M)

        else:
            # main model computation - use _encode_global to properly handle sin/cos
            encoded_state = self._encode_global(x_current_global)  # (B, 3, 3, 256)
            current_substeps = input_substeps if input_substeps is not None else self.input_substeps
            
            accumulated_delta_pred = 0
            # Sub-stepping loop for single column
            for sub_step in range(current_substeps):
                B, H, W, C = encoded_state.shape
                E_current_flat_patch = encoded_state.view(B, H * W, C)
                delta_pred_sub = self.mixer(E_current_flat_patch)  # (B, 256) (actually B*9, but output is 256?) - wait, mixer output depends on input
                # Mixer inputs/outputs are flattened patches or heads?
                # In single col mode, input is (B, 3, 3, M). _extract_patches inside _compute_latent_delta handles padding
                # But here we are calling self.mixer directly?
                # Wait, the original code called `self.mixer(E_current_flat_patch)`.
                # `encoded_state` is (B, 3, 3, 256). `E_current_flat_patch` is (B, 9, 256).
                # `self.mixer` is a Sequential of MixerBlocks.
                # StandardMixerBlock forward expects (Batch, NumPatches, Dim).
                # If we pass (B, 9, 256), it treats B as batch and 9 as num_patches.
                # That seems correct for single column 3x3 patch.
                
                # Apply delta
                center_state = encoded_state[:, 1, 1]  # (B, 256) 
                # !!! Wait, mixer outputs centered column?
                # No, StandardMixerBlock outputs (B, NumPatches, Dim).
                # FinalMixerBlock outputs (B, Dim) (with re-appended sin/cos).
                # If the last block is FinalMixerBlock, it returns (B, 256).
                
                # So delta_pred_sub is (B, 256).
                
                delta_learanble = delta_pred_sub[:, :-2]
                
                # Update center state
                center_learnable_next = center_state[:, :-2] + delta_learanble
                center_sin_cos = center_state[:, -2:]
                center_next = torch.cat([center_learnable_next, center_sin_cos], dim=-1) # (B, 256)
                
                # Now we need to update the FULL 3x3 patch for the next step?
                # In single column mode, we usually assume neighbors are constant or we don't have them?
                # The original code only updated `center_state`.
                # `encoded_state` was just `self._encode_global(x_current_global)`.
                # If we loop, we need a refined 3x3 patch.
                # But we only have usage for 1 column. 
                # If we assume neighbors don't change (Dirichlet/Fixed BCs for neighbors in this test mode), 
                # we can construct the next patch.
                
                # For consistency with the global loop, let's update the center in the patch.
                # But `encoded_state` is (B, 3, 3, 256). 
                # We can update encoded_state[:, 1, 1] = center_next.
                encoded_state[:, 1, 1] = center_next
                
                if isinstance(accumulated_delta_pred, int):
                    accumulated_delta_pred = delta_pred_sub
                else:
                    accumulated_delta_pred = accumulated_delta_pred + delta_pred_sub
            
            delta_pred = accumulated_delta_pred
            
            # Final E_next is the last center_next
            E_next = encoded_state[:, 1, 1] 
            decoded_pred = self.decoder(E_next)

            # calculate auxilary outputs for loss computation
            encoded_next_true = self._encode_global(x_next_global)  # (B, 3, 3, 256)
            encoded_next_true_center = encoded_next_true[:, 1, 1]  # (B, 256)
            delta_true = encoded_next_true_center - center_state  # True delta (sin/cos part should be ~0)
            reconstructed_true = self.decoder(encoded_next_true_center)
            tend_pred = self.tend_decoder(delta_pred)
            tend_true = self.tend_decoder(delta_true)

            return delta_pred, delta_true, decoded_pred, reconstructed_true, tend_pred, tend_true



# ==============================================================================
# === Self-Contained Test Block                                              ===
# ==============================================================================

if __name__ == '__main__':
    # --- User-Selectable Hyperparameters ---
    BATCH_SIZE = 2
    N_LAT = 32   # Using smaller grid for faster testing
    N_LON = 64
    M_FEATURES = 122
    PATCH_SIZE = 3

    MODEL_DIM = 258  # 256 learnable + 2 for sin/cos lat (256 divisible by 16 heads)
    N_HEADS = 16

    NUM_MIXER_BLOCKS = 1 # Must be >= 1
    NUM_CORRECTION_BLOCKS = 0 # Test correction layers
    USE_SPECTRAL_FILTER = False  # Test spectral filter
    INPUT_SUBSTEPS = 2
    
    # --- MixerEncoder Configuration ---
    USE_MIXER_ENCODER = False  # Test MLP-Mixer encoder instead of pointwise MLP
    NUM_ENCODER_MIXER_BLOCKS = 1  # Number of mixer blocks in encoder

    # --- MLP Shape Configurations ---
    HEAD_MLP_HIDDEN_DIMS = [96, 256, 128, 64]
    FFN_HIDDEN_DIMS = [MODEL_DIM * 4, MODEL_DIM * 2]
    ENCODER_HIDDEN_DIMS = [512,512]
    
    CORRECTION_HEAD_MLP_HIDDEN_DIMS = [32,16]
    CORRECTION_FFN_HIDDEN_DIMS = [MODEL_DIM]

    # --- Model Instantiation ---
    print("--- Latent Global Atmos Mixer Configuration ---")
    print(f"Grid size: {N_LAT}x{N_LON}")
    print(f"Input/Output features (M): {M_FEATURES}")
    print(f"Internal model dimension: {MODEL_DIM}")
    print(f"Encoder hidden layers: {ENCODER_HIDDEN_DIMS}")
    print(f"Number of mixer blocks (N-1 standard + 1 final): {NUM_MIXER_BLOCKS}")
    print(f"Number of correction blocks: {NUM_CORRECTION_BLOCKS}")
    print(f"Use spectral filter: {USE_SPECTRAL_FILTER}")
    print(f"Use mixer encoder: {USE_MIXER_ENCODER}")
    print(f"Input substeps: {INPUT_SUBSTEPS}")
    print("-" * 45)

    model = NNorm_LatentGlobalAtmosMixer2(
        M_features=M_FEATURES,
        model_dim=MODEL_DIM,
        n_heads=N_HEADS,
        num_mixer_blocks=NUM_MIXER_BLOCKS,
        head_mlp_hidden_dims=HEAD_MLP_HIDDEN_DIMS,
        ffn_hidden_dims=FFN_HIDDEN_DIMS,
        encoder_hidden_dims=ENCODER_HIDDEN_DIMS,
        patch_size=PATCH_SIZE,
        dropout_rate=0.1,
        num_correction_blocks=NUM_CORRECTION_BLOCKS,
        correction_head_mlp_hidden_dims=CORRECTION_HEAD_MLP_HIDDEN_DIMS,
        correction_ffn_hidden_dims=CORRECTION_FFN_HIDDEN_DIMS,
        use_spectral_filter=USE_SPECTRAL_FILTER,
        n_lat=N_LAT,
        n_lon=N_LON,
        use_sfno=False,  # Test SFNO
        sfno_embed_dim=16,
        sfno_num_layers=1,
        sfno_use_mlp=False,
        sfno_scale_factor=2,
        # MixerEncoder params
        use_mixer_encoder=USE_MIXER_ENCODER,
        num_encoder_mixer_blocks=NUM_ENCODER_MIXER_BLOCKS,
        input_substeps=INPUT_SUBSTEPS,
    )

    # --- Print Encoder Parameters ---
    if model.use_mixer_encoder:
        enc_params = sum(p.numel() for p in model.encoder_input_proj.parameters())
        enc_params += sum(p.numel() for p in model.encoder_mixer_blocks.parameters())
        print(f"\nEncoder (MixerEncoder) parameters: {enc_params:,}")
    else:
        enc_params = sum(p.numel() for p in model.encoder.parameters())
        print(f"\nEncoder (MLP) parameters: {enc_params:,}")
    
    # Latent mixer parameters (main mixer blocks, excluding encoder and decoders)
    mixer_params = sum(p.numel() for p in model.mixer.parameters())
    print(f"Latent Mixer parameters: {mixer_params:,}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    # --- Create Dummy Data ---
    dummy_x_current = torch.randn(BATCH_SIZE, N_LAT, N_LON, M_FEATURES)
    dummy_x_next = torch.randn(BATCH_SIZE, N_LAT, N_LON, M_FEATURES)

    print(f"\nInput tensor shapes (current/next): {dummy_x_current.shape}")

    # --- Perform a Forward Pass ---
    # Model returns 8 values in training mode: delta_pred, delta_true, x_next_pred, reconstructed_true, 
    # tend_pred, tend_true, final_pred_latent, final_true_latent
    # Model returns 8 values in training mode: delta_pred, delta_true, x_next_pred, reconstructed_true, 
    # tend_pred, tend_true, final_pred_latent, final_true_latent
    print(f"\nRunning forward pass with input_substeps={INPUT_SUBSTEPS}...")
    outputs = model(dummy_x_current, dummy_x_next, input_substeps=INPUT_SUBSTEPS, num_rollout_steps=1)
    delta_E_pred, delta_E_true, x_next_pred, reconstructed_true, tend_pred, tend_true, final_pred_latent, final_true_latent = outputs

    print("\n--- Output Shapes ---")
    print(f"Predicted Latent Tendency (delta_E_pred): {delta_E_pred.shape}")
    print(f"  > Expected: {(BATCH_SIZE, 1, N_LAT, N_LON, MODEL_DIM)}")
    print(f"True Latent Tendency (delta_E_true):      {delta_E_true.shape}")
    print(f"  > Expected: {(BATCH_SIZE, 1, N_LAT, N_LON, MODEL_DIM)}")
    print(f"Decoded Physical Prediction (x_next_pred): {x_next_pred.shape}")
    print(f"  > Expected: {(BATCH_SIZE, 1, N_LAT, N_LON, M_FEATURES)}")


    # --- Calculate and Print Total Parameters ---
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")

# %%

# %%
    from torch_harmonics.examples.models import SphericalFourierNeuralOperator
    grid = "equiangular"
    model = SphericalFourierNeuralOperator(img_size=(192, 288), in_chans=256, out_chans=256, encoder_layers = 3, grid=grid, num_layers=4, scale_factor=10, embed_dim=256, residual_prediction=True, pos_embed="spectral", use_mlp=True, mlp_ratio=1, normalization_layer="none")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

# %%
    sfno_filter = SphericalFourierNeuralOperator(
    in_chans=256,
    out_chans=256,
    embed_dim=64,
    num_layers=3,              # Single spectral layer! 
    use_mlp=True, 
    mlp_ratio=4,
    residual_prediction=True,
    scale_factor=1,
    pos_embed="learnable lat"
    )
    total_params = sum(p.numel() for p in sfno_filter.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
# %%
    sfno_filter = SphericalFourierNeuralOperator(
    in_chans=7,
    out_chans=6,
    embed_dim=72,
    num_layers=8,              # Single spectral layer! 
    use_mlp=True,   
    mlp_ratio =4,
    residual_prediction=True,
    scale_factor=1,
    )
    total_params = sum(p.numel() for p in sfno_filter.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    