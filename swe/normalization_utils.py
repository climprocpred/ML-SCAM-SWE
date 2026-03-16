"""
Normalization utilities for SWE training.

Computes stable normalization statistics from EQUILIBRATED states (not raw ICs)
by spinning up the solver before sampling. This ensures stats match the distribution
the model will see during long autoregressive rollouts.

Cache key includes: nlat, nlon, grid, mach, n_samples, spinup_steps
"""

import os
import torch
from tqdm import tqdm


def get_cache_filename(nlat: int, nlon: int, grid: str, mach: float, 
                       n_samples: int, spinup_steps: int) -> str:
    """Generate cache filename based on parameters."""
    return f"norm_stats_{nlat}x{nlon}_{grid}_mach{mach}_n{n_samples}_spinup{spinup_steps}_v2.pt"


def compute_normalization_stats(
    solver,
    n_samples: int = 100,
    mach: float = 0.2,
    spinup_steps: int = 100,
    dt: int = 600,
    dt_solver: int = 150,
    show_progress: bool = True
) -> dict:
    """
    Compute normalization statistics from equilibrated SWE states.
    
    For each sample:
    1. Generate random IC
    2. Run solver for `spinup_steps` output timesteps to reach equilibrium
    3. Sample statistics from the equilibrated state
    
    Parameters
    ----------
    solver : ShallowWaterSolver
        The SWE solver instance
    n_samples : int
        Number of equilibrated states to sample
    mach : float
        Mach number for random IC generation
    spinup_steps : int
        Number of output timesteps to run before sampling (for equilibration)
    dt : int
        Output time step in seconds
    dt_solver : int
        Internal solver time step in seconds
    show_progress : bool
        Whether to show progress bar
        
    Returns
    -------
    dict with keys 'mean', 'var', 'uv_mean', 'uv_var'
    """
    device = solver.lap.device
    solver_steps_per_output = dt // dt_solver
    
    # Warn about CPU instability
    if device.type == 'cpu':
        print("WARNING: SWE solver may be unstable on CPU with float32. "
              "Stats computation is recommended on GPU.")
    
    # Accumulators for online mean/variance (Welford's algorithm)
    count = 0
    mean = None
    M2 = None  # For variance of means
    var_sum = None  # For averaging spatial variances
    
    # UV Accumulators
    uv_mean = None
    uv_M2 = None
    uv_var_sum = None
    
    nan_count = 0
    
    iterator = range(n_samples)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Computing stats (spinup={spinup_steps} steps)")
    
    for i in iterator:
        # Generate random IC
        ic_spec = solver.random_initial_condition(mach=mach)
        
        # Spin up to equilibrium
        current_spec = ic_spec
        is_valid = True
        for step in range(spinup_steps):
            current_spec = solver.timestep(current_spec, solver_steps_per_output)
            if torch.isnan(current_spec).any():
                print(f"WARNING: NaN detected in sample {i} at spinup step {step+1}/{spinup_steps}")
                is_valid = False
                nan_count += 1
                break
        
        if not is_valid:
            continue  # Skip this sample
        
        # Convert equilibrated state to grid space
        equilibrated_grid = solver.spec2grid(current_spec)  # Shape: (C, nlat, nlon)
        
        if torch.isnan(equilibrated_grid).any():
            print(f"WARNING: NaN detected in sample {i} after grid conversion")
            nan_count += 1
            continue
            
        # Compute U and V fields from spectral coefficients
        # current_spec contains (height, vorticity, divergence)
        # We need to compute (u, v) from (vorticity, divergence) which are indices 1:
        uv_grid = solver.getuv(current_spec[1:])  # Shape: (2, nlat, nlon) where 0=u, 1=v
        
        # Per-channel spatial mean/var for State variables (H, Z, D)
        sample_mean = equilibrated_grid.mean(dim=(-1, -2))  # Shape: (C,)
        sample_var = equilibrated_grid.var(dim=(-1, -2))    # Shape: (C,)
        
        # Per-channel spatial mean/var for UV variables (U, V)
        uv_sample_mean = uv_grid.mean(dim=(-1, -2))  # Shape: (2,)
        uv_sample_var = uv_grid.var(dim=(-1, -2))    # Shape: (2,)
        
        count += 1
        
        if mean is None:
            # Initialize State stats
            mean = sample_mean.clone()
            M2 = torch.zeros_like(sample_mean)
            var_sum = sample_var.clone()
            
            # Initialize UV stats
            uv_mean = uv_sample_mean.clone()
            uv_M2 = torch.zeros_like(uv_sample_mean)
            uv_var_sum = uv_sample_var.clone()
        else:
            # Welford's online update for State mean
            delta = sample_mean - mean
            mean = mean + delta / count
            M2 = M2 + delta * (sample_mean - mean)
            # Simple average for State spatial variance
            var_sum = var_sum + sample_var
            
            # Welford's online update for UV mean
            uv_delta = uv_sample_mean - uv_mean
            uv_mean = uv_mean + uv_delta / count
            uv_M2 = uv_M2 + uv_delta * (uv_sample_mean - uv_mean)
            # Simple average for UV spatial variance
            uv_var_sum = uv_var_sum + uv_sample_var
    
    if nan_count > 0:
        print(f"WARNING: {nan_count}/{n_samples} samples produced NaN and were skipped")
    
    if count == 0:
        raise RuntimeError("All samples produced NaN. Try running on GPU or using double precision.")
    
    # Final statistics
    spatial_var = var_sum / count
    uv_spatial_var = uv_var_sum / count
    
    return {
        "mean": mean.reshape(-1, 1, 1),
        "var": spatial_var.reshape(-1, 1, 1),
        "uv_mean": uv_mean.reshape(-1, 1, 1),
        "uv_var": uv_spatial_var.reshape(-1, 1, 1),
        "n_samples": n_samples,
        "mach": mach,
        "spinup_steps": spinup_steps,
    }


def save_stats(stats: dict, path: str) -> None:
    """Save normalization stats to disk."""
    torch.save(stats, path)
    print(f"Saved normalization stats to {path}")


def load_stats(path: str, device: torch.device) -> dict:
    """Load normalization stats from disk."""
    stats = torch.load(path, map_location=device, weights_only=False)
    print(f"Loaded normalization stats from {path}")
    return stats


def get_or_compute_stats(
    solver,
    cache_dir: str,
    n_samples: int = 100,
    mach: float = 0.2,
    spinup_steps: int = 100,
    dt: int = 600,
    dt_solver: int = 150,
    show_progress: bool = True
) -> dict:
    """
    Get normalization stats from cache, or compute and cache them.
    
    Uses EQUILIBRATED states (after spinup) rather than raw ICs. This ensures
    the stats match what the model will see during long autoregressive rollouts.
    
    Cache filename includes all relevant parameters, so changing any of them
    automatically creates a new cache.
    
    Parameters
    ----------
    solver : ShallowWaterSolver
        The SWE solver instance
    cache_dir : str
        Directory to store cache files
    n_samples : int
        Number of equilibrated states to sample
    mach : float
        Mach number for random IC generation
    spinup_steps : int
        Number of output timesteps for equilibration (default: 100)
    dt : int
        Output time step in seconds
    dt_solver : int
        Internal solver time step
    show_progress : bool
        Whether to show progress bar during computation
        
    Returns
    -------
    dict with keys 'mean', 'var' as tensors of shape (C, 1, 1)
    """
    nlat = solver.nlat
    nlon = solver.nlon
    grid = solver.grid
    device = solver.lap.device
    
    # Create cache directory if needed
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate parameter-aware cache filename
    cache_filename = get_cache_filename(nlat, nlon, grid, mach, n_samples, spinup_steps)
    cache_path = os.path.join(cache_dir, cache_filename)
    
    # Check for cached stats
    if os.path.exists(cache_path):
        stats = load_stats(cache_path, device)
        stats["mean"] = stats["mean"].to(device)
        stats["var"] = stats["var"].to(device)
        return stats
    
    # Compute new stats
    print(f"Computing normalization stats from {n_samples} equilibrated samples...")
    print(f"  Grid: {nlat}x{nlon} {grid}, Mach: {mach}")
    print(f"  Spinup: {spinup_steps} steps ({spinup_steps * dt / 3600:.1f} hours)")
    
    stats = compute_normalization_stats(
        solver, n_samples, mach, spinup_steps, dt, dt_solver, show_progress
    )
    
    # Save to cache
    save_stats(stats, cache_path)
    
    return stats
