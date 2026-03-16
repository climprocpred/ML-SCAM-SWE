# coding=utf-8
"""
Conservation Loss Module

Provides a loss function that penalizes changes in the area-weighted global mean
for each variable between input and output. This enforces:
- Mass conservation (height/geopotential)
- Circulation conservation (vorticity)
- Zero global divergence (divergence)
"""

import torch
import torch.nn as nn
from torch_harmonics.quadrature import clenshaw_curtiss_weights, legendre_gauss_weights


class ConservationLossS2(nn.Module):
    """
    Loss that penalizes changes in the area-weighted global mean for each channel.
    
    For SWE with channels [height, vorticity, divergence]:
    - Height: mass conservation (global mean should be constant)
    - Vorticity: circulation conservation (global mean should be constant)
    - Divergence: should integrate to zero (global mean should stay zero)
    
    L = sum_c || <pred_c> - <input_c> ||^2
    
    where <.> denotes area-weighted mean.
    """
    
    def __init__(
        self,
        nlat: int,
        nlon: int,
        grid: str = "equiangular",
    ):
        """
        Args:
            nlat: Number of latitude points
            nlon: Number of longitude points  
            grid: Grid type ("equiangular" or "legendre-gauss")
        """
        super().__init__()
        
        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        
        # Compute quadrature weights for proper spherical integration
        if grid == "equiangular":
            _, weights = clenshaw_curtiss_weights(nlat, -1, 1)
        elif grid == "legendre-gauss":
            _, weights = legendre_gauss_weights(nlat, -1, 1)
        else:
            raise ValueError(f"Unknown grid type: {grid}")
        
        # Convert to tensor
        # Shape: (nlat,)
        if isinstance(weights, torch.Tensor):
            weights = weights.clone().detach().float()
        else:
            weights = torch.tensor(weights, dtype=torch.float32)
        
        # Normalize weights to sum to 1 (for proper mean computation)
        weights = weights / weights.sum()
        self.register_buffer('weights', weights)
    
    def compute_global_mean(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the area-weighted global mean for each channel.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Global means of shape (B, C)
        """
        # Average over longitude first (uniform weighting)
        x_lon_mean = x.mean(dim=-1)  # (B, C, H)
        
        # Weighted average over latitude
        # weights shape: (H,), broadcast to (1, 1, H)
        weighted = x_lon_mean * self.weights.view(1, 1, -1)
        global_mean = weighted.sum(dim=-1)  # (B, C)
        
        return global_mean
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the conservation loss between prediction and target.
        
        Since the physical solver conserves global means, the target has the same
        global mean as the initial condition. We penalize the prediction for
        deviating from the target's global mean.
        
        Args:
            pred: Prediction tensor of shape (B, C, H, W)
            target: Target tensor of shape (B, C, H, W)
            
        Returns:
            Scalar loss value (mean over batch and channels)
        """
        # Compute global means
        pred_mean = self.compute_global_mean(pred)      # (B, C)
        target_mean = self.compute_global_mean(target)  # (B, C)
        
        # Squared difference in global means
        loss = (pred_mean - target_mean) ** 2
        
        # Mean over batch and channels
        return loss.mean()
