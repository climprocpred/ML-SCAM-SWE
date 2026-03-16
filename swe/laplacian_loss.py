# coding=utf-8
"""
Laplacian Loss Module

Provides a loss function that penalizes differences in the spatial Laplacian
between predictions and ground truth on the sphere.

The spherical Laplacian is computed efficiently in spectral space using the
relationship: Laplacian(Y_l^m) = -l(l+1) * Y_l^m
"""

import torch
import torch.nn as nn
from torch_harmonics import RealSHT, InverseRealSHT
from torch_harmonics.quadrature import clenshaw_curtiss_weights, legendre_gauss_weights


class LaplacianLossS2(nn.Module):
    """
    Loss that penalizes the difference between the spatial Laplacian of
    predictions and ground truth on the sphere.
    
    The Laplacian is computed in spectral space where it becomes a simple
    multiplication by -l(l+1). This makes it efficient and exact.
    
    L = || Laplacian(pred) - Laplacian(target) ||^2_S2
    
    where ||.||_S2 is the L2 norm weighted by the spherical quadrature weights.
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
            normalize: If True, normalize by the sum of quadrature weights
        """
        super().__init__()
        
        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.normalize = normalize
        
        # Set lmax (default to nlat if not specified)
        if lmax is None:
            lmax = nlat
        self.lmax = lmax
        # mmax is constrained by the longitude grid: max order is nlon // 2 + 1
        mmax = min(lmax, nlon // 2 + 1)
        self.mmax = mmax
        
        # Initialize SHT and inverse SHT
        self.sht = RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid)
        self.isht = InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid)
        
        # Compute Laplacian eigenvalues: -l(l+1) for each (l, m)
        # Shape: (lmax, mmax) to match SHT output
        l = torch.arange(0, lmax, dtype=torch.float32)
        lap_eigenvalues = -l * (l + 1)
        # Expand to (lmax, mmax) - eigenvalue depends only on l, not m
        lap_eigenvalues = lap_eigenvalues.unsqueeze(1).expand(lmax, mmax)
        self.register_buffer('lap_eigenvalues', lap_eigenvalues)
        
        # Compute quadrature weights for proper spherical integration
        if grid == "equiangular":
            _, weights = clenshaw_curtiss_weights(nlat, -1, 1)
        elif grid == "legendre-gauss":
            _, weights = legendre_gauss_weights(nlat, -1, 1)
        else:
            raise ValueError(f"Unknown grid type: {grid}")
        
        # Convert to tensor and scale for longitude integration
        # Shape: (nlat,)
        if isinstance(weights, torch.Tensor):
            weights = weights.clone().detach().float()
        else:
            weights = torch.tensor(weights, dtype=torch.float32)
        weights = 2.0 * 3.14159265358979 / nlon * weights
        self.register_buffer('weights', weights)
    
    def _compute_laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the spatial Laplacian of x in spectral space.
        
        Args:
            x: Input tensor of shape (B, C, H, W) or (B, H, W)
            
        Returns:
            Laplacian of x with same shape as input
        """
        orig_shape = x.shape
        
        # Handle (B, C, H, W) by flattening batch and channel
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.reshape(B * C, H, W)
        elif x.dim() == 3:
            pass  # Already (B, H, W)
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D")
        
        # Transform to spectral space
        x_spec = self.sht(x)  # (B*C, lmax, mmax) complex
        
        # Apply Laplacian eigenvalues
        lap_spec = x_spec * self.lap_eigenvalues
        
        # Transform back to grid space
        lap_grid = self.isht(lap_spec)  # (B*C, H, W)
        
        # Reshape back to original shape
        if len(orig_shape) == 4:
            B, C, H, W = orig_shape
            lap_grid = lap_grid.reshape(B, C, H, W)
        
        return lap_grid
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Laplacian loss between prediction and target.
        
        Args:
            pred: Prediction tensor of shape (B, C, H, W)
            target: Target tensor of shape (B, C, H, W)
            
        Returns:
            Scalar loss value (mean over batch and channels)
        """
        # Compute Laplacians
        lap_pred = self._compute_laplacian(pred)
        lap_target = self._compute_laplacian(target)
        
        # Compute squared difference
        diff_sq = (lap_pred - lap_target) ** 2  # (B, C, H, W)
        
        # Sum over longitude
        diff_sq_sum_lon = diff_sq.sum(dim=-1)  # (B, C, H)
        
        # Apply quadrature weights for proper spherical integration
        # weights shape: (H,), broadcast to (1, 1, H)
        weighted = diff_sq_sum_lon * self.weights.view(1, 1, -1)
        
        # Sum over latitude
        loss = weighted.sum(dim=-1)  # (B, C)
        
        # Normalize if requested
        if self.normalize:
            loss = loss / self.weights.sum()
        
        # Mean over batch and channels
        return loss.mean()


class SquaredLaplacianLossS2(LaplacianLossS2):
    """Alias for LaplacianLossS2 for consistency with SquaredL2LossS2 naming."""
    pass
