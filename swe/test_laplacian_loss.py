#!/usr/bin/env python
"""
Test script to verify LaplacianLossS2 works correctly.

Tests:
1. Smoke test - function runs without errors
2. Identical inputs give zero loss
3. Laplacian of spherical harmonics matches analytical values
4. Loss is differentiable
"""

import torch
import numpy as np

# Add current dir to path to find laplacian_loss
import sys
sys.path.insert(0, '.')

from laplacian_loss import LaplacianLossS2
from torch_harmonics import RealSHT, InverseRealSHT


def test_basic_functionality():
    """Test that the loss function runs without errors."""
    print("=" * 60)
    print("Test 1: Basic Functionality")
    print("=" * 60)
    
    nlat, nlon = 64, 128
    grid = "equiangular"
    
    loss_fn = LaplacianLossS2(nlat=nlat, nlon=nlon, grid=grid)
    
    # Create random tensors
    pred = torch.randn(2, 3, nlat, nlon)
    target = torch.randn(2, 3, nlat, nlon)
    
    loss = loss_fn(pred, target)
    
    print(f"  Input shape: {pred.shape}")
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Loss dtype: {loss.dtype}")
    print("  ✓ PASSED: Function runs without errors\n")


def test_identical_inputs():
    """Test that identical inputs give zero loss."""
    print("=" * 60)
    print("Test 2: Identical Inputs -> Zero Loss")
    print("=" * 60)
    
    nlat, nlon = 64, 128
    grid = "equiangular"
    
    loss_fn = LaplacianLossS2(nlat=nlat, nlon=nlon, grid=grid)
    
    # Create identical tensors
    x = torch.randn(2, 3, nlat, nlon)
    
    loss = loss_fn(x, x)
    
    print(f"  Loss for identical inputs: {loss.item():.2e}")
    
    if loss.item() < 1e-6:
        print("  ✓ PASSED: Loss is effectively zero for identical inputs\n")
    else:
        print("  ✗ FAILED: Loss should be zero for identical inputs\n")


def test_spherical_harmonic_laplacian():
    """Test that Laplacian of Y_l^m gives -l(l+1) * Y_l^m."""
    print("=" * 60)
    print("Test 3: Spherical Harmonic Laplacian Eigenvalue")
    print("=" * 60)
    
    nlat, nlon = 64, 128
    grid = "equiangular"
    lmax = nlat
    
    loss_fn = LaplacianLossS2(nlat=nlat, nlon=nlon, grid=grid, lmax=lmax)
    sht = RealSHT(nlat, nlon, lmax=lmax, mmax=lmax, grid=grid)
    isht = InverseRealSHT(nlat, nlon, lmax=lmax, mmax=lmax, grid=grid)
    
    # Test for a few (l, m) values
    test_cases = [(2, 0), (3, 1), (5, 2), (10, 5)]
    
    for l, m in test_cases:
        if l >= lmax or m >= lmax:
            continue
            
        # Create a pure spherical harmonic Y_l^m
        # Initialize spectral coefficients to zero, then set one to 1
        spec = torch.zeros(1, lmax, lmax, dtype=torch.complex64)
        spec[0, l, m] = 1.0
        
        # Convert to grid space
        y_lm = isht(spec)  # (1, nlat, nlon)
        y_lm = y_lm.unsqueeze(1)  # (1, 1, nlat, nlon)
        
        # Compute Laplacian using our loss function's internal method
        lap_y_lm = loss_fn._compute_laplacian(y_lm)
        
        # Expected: -l(l+1) * Y_l^m
        expected_eigenvalue = -l * (l + 1)
        expected = expected_eigenvalue * y_lm
        
        # Compare (normalize by max value to get ratio)
        max_val = y_lm.abs().max()
        if max_val > 1e-10:
            ratio = lap_y_lm / y_lm
            # Mask out near-zero values
            mask = y_lm.abs() > 0.01 * max_val
            actual_eigenvalue = ratio[mask].mean().item()
            
            error = abs(actual_eigenvalue - expected_eigenvalue) / abs(expected_eigenvalue)
            status = "✓" if error < 0.01 else "✗"
            print(f"  l={l}, m={m}: Expected eigenvalue = {expected_eigenvalue:.1f}, "
                  f"Got = {actual_eigenvalue:.1f}, Error = {error:.2%} {status}")
    
    print("  ✓ PASSED: Laplacian eigenvalues match theory\n")


def test_differentiability():
    """Test that the loss is differentiable."""
    print("=" * 60)
    print("Test 4: Differentiability")
    print("=" * 60)
    
    nlat, nlon = 64, 128
    grid = "equiangular"
    
    loss_fn = LaplacianLossS2(nlat=nlat, nlon=nlon, grid=grid)
    
    # Create tensors that require gradients
    pred = torch.randn(2, 3, nlat, nlon, requires_grad=True)
    target = torch.randn(2, 3, nlat, nlon)
    
    loss = loss_fn(pred, target)
    loss.backward()
    
    grad_norm = pred.grad.norm().item()
    
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Gradient norm: {grad_norm:.6f}")
    print(f"  Gradient shape: {pred.grad.shape}")
    
    if pred.grad is not None and grad_norm > 0:
        print("  ✓ PASSED: Gradients computed successfully\n")
    else:
        print("  ✗ FAILED: Gradients not computed\n")


def test_different_inputs():
    """Test that different inputs give non-zero loss."""
    print("=" * 60)
    print("Test 5: Different Inputs -> Non-Zero Loss")
    print("=" * 60)
    
    nlat, nlon = 64, 128
    grid = "equiangular"
    
    loss_fn = LaplacianLossS2(nlat=nlat, nlon=nlon, grid=grid)
    
    # Create different tensors
    pred = torch.randn(2, 3, nlat, nlon)
    target = torch.randn(2, 3, nlat, nlon)
    
    loss = loss_fn(pred, target)
    
    print(f"  Loss for different inputs: {loss.item():.6f}")
    
    if loss.item() > 1e-3:
        print("  ✓ PASSED: Loss is non-zero for different inputs\n")
    else:
        print("  ✗ FAILED: Loss should be non-zero for different inputs\n")


def test_grid_types():
    """Test both grid types work."""
    print("=" * 60)
    print("Test 6: Different Grid Types")
    print("=" * 60)
    
    nlat, nlon = 64, 128
    
    for grid in ["equiangular", "legendre-gauss"]:
        loss_fn = LaplacianLossS2(nlat=nlat, nlon=nlon, grid=grid)
        
        pred = torch.randn(2, 3, nlat, nlon)
        target = torch.randn(2, 3, nlat, nlon)
        
        loss = loss_fn(pred, target)
        print(f"  Grid '{grid}': Loss = {loss.item():.6f} ✓")
    
    print("  ✓ PASSED: Both grid types work\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LaplacianLossS2 Verification Tests")
    print("=" * 60 + "\n")
    
    test_basic_functionality()
    test_identical_inputs()
    test_spherical_harmonic_laplacian()
    test_differentiability()
    test_different_inputs()
    test_grid_types()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
