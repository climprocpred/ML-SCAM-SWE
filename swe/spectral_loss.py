# coding=utf-8
"""
Spectral Loss Module

Provides a loss function that directly penalizes the spectral power of
predictions vs ground truth, allowing for targeted damping of specific 
wavenumber bands (e.g., high frequencies).
"""

import torch
import torch.nn as nn
from torch_harmonics import RealSHT

class SpectralLossS2(nn.Module):
    """
    Loss that penalizes the difference in spectral coefficients, with 
    configurable weighting by wavenumber l.
    
    L = sum_{l,m} w_l * |c_{lm}^{pred} - c_{lm}^{target}|^2
    
    where w_l = l^decay_exponent if l in [lmin, lmax], else 0.
    """
    
    def __init__(
        self,
        nlat: int,
        nlon: int,
        grid: str = "equiangular",
        lmax: int = None,
        lmin_loss: int = 0,
        lmax_loss: int = None,
        decay_exponent: float = 0.0,
        normalize: bool = True,
    ):
        """
        Args:
            nlat: Number of latitude points
            nlon: Number of longitude points  
            grid: Grid type ("equiangular" or "legendre-gauss")
            lmax: Maximum spherical harmonic degree of SHT (default: nlat)
            lmin_loss: Minimum wavenumber l to include in loss (default: 0)
            lmax_loss: Maximum wavenumber l to include in loss (default: lmax)
            decay_exponent: Exponent k for weighting term l^k (default: 0)
                            k=0: Uniform weighting
                            k=2: Similar to Laplacian (penalizes curvature)
                            k>2: Stronger penalty on high frequencies
            normalize: If True, normalize by the number of active modes
        """
        super().__init__()
        
        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        
        # Set lmax (default to nlat if not specified)
        if lmax is None:
            lmax = nlat
        self.lmax = lmax
        # mmax is constrained by the longitude grid
        mmax = min(lmax, nlon // 2 + 1)
        self.mmax = mmax
        
        # Loss range
        self.lmin_loss = lmin_loss
        self.lmax_loss = lmax_loss if lmax_loss is not None else lmax
        self.decay_exponent = decay_exponent
        self.normalize = normalize
        
        # Initialize SHT
        self.sht = RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid)
        
        # Precompute weights: (lmax, mmax)
        # Weight depends only on l
        l = torch.arange(0, lmax, dtype=torch.float32)
        
        # Mask for valid range [lmin_loss, lmax_loss]
        mask = (l >= self.lmin_loss) & (l <= self.lmax_loss)
        
        # Weighting term l^k
        weights = torch.pow(l, self.decay_exponent)
        
        # Apply mask
        weights = weights * mask.float()
        
        # Expand to (lmax, mmax)
        weights = weights.unsqueeze(1).expand(lmax, mmax)
        self.register_buffer('weights', weights)
        
        # Count active modes for normalization
        # Each (l,m) with m>0 corresponds to 2 real degrees of freedom (real/imag), m=0 is 1.
        # But here we just normalize by sum of weights to keep scale consistent
        self.weight_sum = weights.sum() if normalize else 1.0

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Spectral loss.
        
        Args:
            pred: Prediction tensor of shape (B, C, H, W)
            target: Target tensor of shape (B, C, H, W)
            
        Returns:
            Scalar loss value
        """
        orig_shape = pred.shape
        
        # Handle (B, C, H, W) -> (B*C, H, W)
        if pred.dim() == 4:
            B, C, H, W = pred.shape
            pred = pred.reshape(B * C, H, W)
            target = target.reshape(B * C, H, W)
            
        # Transform to spectral space
        pred_spec = self.sht(pred)      # (B*C, lmax, mmax) complex
        target_spec = self.sht(target)  # (B*C, lmax, mmax) complex
        
        # Squared difference of coefficients
        diff_sq = (pred_spec - target_spec).abs().pow(2)
        
        # Apply spectral weights
        weighted_diff = diff_sq * self.weights
        
        # Sum over all modes (l, m) and batch/channel
        loss = weighted_diff.sum()
        
        # Normalize
        if self.normalize and self.weight_sum > 0:
            loss = loss / (self.weight_sum * (B * C))
        else:
            loss = loss / (B * C)
            
        return loss

class GradientLossS2(SpectralLossS2):
    """
    Loss that penalizes the gradient norm (H1 semi-norm) of the difference
    between predictions and ground truth.
    
    Equivalent to SpectralLossS2 with weights w_l = l(l+1).
    This corresponds to the integral of |grad(pred - target)|^2 on the sphere.
    """
    
    def __init__(
        self,
        nlat: int,
        nlon: int,
        grid: str = "equiangular",
        lmax: int = None,
        normalize: bool = True,
    ):
        """
        Args:
            nlat: Number of latitude points
            nlon: Number of longitude points  
            grid: Grid type ("equiangular" or "legendre-gauss")
            lmax: Maximum spherical harmonic degree (default: nlat)
            normalize: If True, normalize by the sum of weights
        """
        # Initialize SpectralLossS2 with dummy values, we'll overwrite weights
        super().__init__(nlat, nlon, grid=grid, lmax=lmax, normalize=normalize)
        
        # Override weights with l(l+1)
        l = torch.arange(0, self.lmax, dtype=torch.float32)
        weights = l * (l + 1)
        
        # Expand to (lmax, mmax)
        weights = weights.unsqueeze(1).expand(self.lmax, self.mmax)
        self.register_buffer('weights', weights)
        
        # Recalculate weight sum
        self.weight_sum = weights.sum() if normalize else 1.0
