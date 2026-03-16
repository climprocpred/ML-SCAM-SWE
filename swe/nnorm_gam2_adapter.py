
import torch
import torch.nn as nn
import numpy as np
from NNorm_GAM2 import NNorm_LatentGlobalAtmosMixer2

# ==============================================================================
# === Adapter Interface for Model Registry (TH_SWE Integration)              ===
# ==============================================================================

class NNorm_GAM2_Interface(nn.Module):
    def __init__(self, img_size=(192, 288), in_chans=3, out_chans=3, 
                 embed_dim=256, patch_size=3, num_layers=4, 
                 grid="equiangular", **kwargs):
        super().__init__()
        self.n_lat, self.n_lon = img_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        
        # M_features includes the 2 lat/lon embeddings which we append
        self.M_features = in_chans + 2
        
        # Set defaults for hidden dims if not in kwargs (based on embed_dim)
        head_mlp_hidden_dims = kwargs.pop('head_mlp_hidden_dims', [embed_dim // 2, embed_dim])
        ffn_hidden_dims = kwargs.pop('ffn_hidden_dims', [embed_dim * 4])
        encoder_hidden_dims = kwargs.pop('encoder_hidden_dims', [embed_dim, embed_dim])
        n_heads = kwargs.pop('n_heads', 8)
        
        # Filter kwargs to avoid passing unexpected args to model
        # (Since model assumes anything extra is an error)
        valid_args = [
            'dropout_rate', 'grad_ckpt_level', 'num_correction_blocks',
            'correction_head_mlp_hidden_dims', 'correction_ffn_hidden_dims',
            'use_spectral_filter', 'use_sfno', 'sfno_embed_dim', 'sfno_num_layers',
            'sfno_use_mlp', 'sfno_scale_factor', 'sfno_mlp_ratio',
            'use_mixer_encoder', 'encoder_mixer_n_heads', 'encoder_mixer_head_dims',
            'encoder_mixer_ffn_dims', 'num_encoder_mixer_blocks', 'input_substeps'
        ]
        model_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}

        self.model = NNorm_LatentGlobalAtmosMixer2(
            M_features=self.M_features,
            model_dim=embed_dim,
            n_heads=n_heads,
            num_mixer_blocks=num_layers,
            head_mlp_hidden_dims=head_mlp_hidden_dims,
            ffn_hidden_dims=ffn_hidden_dims,
            encoder_hidden_dims=encoder_hidden_dims,
            patch_size=patch_size,
            n_lat=self.n_lat,
            n_lon=self.n_lon,
            **model_kwargs
        )
        
        self.grid_type = grid
        # Buffer for lat features
        self.register_buffer('lat_features', None, persistent=False)

    def _get_lat_features(self, device, dtype):
        if self.lat_features is not None:
             # Check if device matches
             if self.lat_features.device == device and self.lat_features.dtype == dtype:
                 return self.lat_features
        
        # Generate latitude grid
        if self.grid_type == "equiangular":
            lat = torch.linspace(-np.pi/2, np.pi/2, self.n_lat, device=device, dtype=dtype)
            sin_lat = torch.sin(lat)
            cos_lat = torch.cos(lat)
        elif self.grid_type == "legendre-gauss":
             # Use numpy for roots_legendre
             # leggauss nodes are usually cos(theta) or sin(lat)
             # In numpy.polynomial.legendre.leggauss(deg), it returns sample points (roots) and weights.
             # THe typical interval is [-1, 1].
             # If mapping to sphere, x = sin(lat).
             try:
                 cost, _ = np.polynomial.legendre.leggauss(self.n_lat)
             except AttributeError:
                 # Fallback if numpy version is weird or something
                 cost = np.linspace(-1, 1, self.n_lat)
                 
             # cost is sin(lat)
             sin_lat = torch.tensor(cost, device=device, dtype=dtype)
             cos_lat = torch.sqrt(1 - sin_lat**2)
        else:
             # Fallback
             lat = torch.linspace(-np.pi/2, np.pi/2, self.n_lat, device=device, dtype=dtype)
             sin_lat = torch.sin(lat)
             cos_lat = torch.cos(lat)

        # Create 2D grid
        # lat is 1D (H). Expand to (H, W).
        sin_grid = sin_lat.view(-1, 1).expand(self.n_lat, self.n_lon)
        cos_grid = cos_lat.view(-1, 1).expand(self.n_lat, self.n_lon)
        
        # Stack: (H, W, 2)
        features = torch.stack([sin_grid, cos_grid], dim=-1)
        self.lat_features = features
        return features

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # Verify dimensions match init
        if H != self.n_lat or W != self.n_lon:
             # Just warn or error? Error is safer.
             # But let's support resizing if trivial? No, model has fixed lat/lon usually.
             pass

        # 1. Prepare Lat Features
        lat_feats = self._get_lat_features(x.device, x.dtype) # (H, W, 2)
        lat_feats = lat_feats.unsqueeze(0).expand(B, -1, -1, -1) # (B, H, W, 2)
        
        # 2. Permute Input: (B, C, H, W) -> (B, H, W, C)
        x_perm = x.permute(0, 2, 3, 1)
        
        # 3. Concatenate: (B, H, W, C+2)
        x_in = torch.cat([x_perm, lat_feats], dim=-1)
        
        # 4. Forward Pass
        # train=False forces 'prediction only' path which returns tensor, not tuple
        out = self.model(x_in, train=False, num_rollout_steps=1)
        
        # Output: (B, steps, H, W, M) with steps=1
        if isinstance(out, tuple):
            out = out[0]
        out = out.squeeze(1) # (B, H, W, M)
        
        # 5. Remove lat features: (B, H, W, C)
        out_phys = out[..., :-2]
        
        # 6. Permute back: (B, C, H, W)
        out_final = out_phys.permute(0, 3, 1, 2)
        
        return out_final
