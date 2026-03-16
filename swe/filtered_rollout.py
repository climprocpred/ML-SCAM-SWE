#%%
# Filtered ML Rollout Script
# Tests if spectral filtering between prediction steps stabilizes autoregressive inference
# Applies the same spectral truncation (lmax = nlat/3) that the SWE solver uses

import numpy as np
import torch
import xarray as xr
import matplotlib.pyplot as plt
from math import ceil
import os

from torch_harmonics import RealSHT, InverseRealSHT
from torch_harmonics.examples.shallow_water_equations import ShallowWaterSolver

try:
    import cartopy.crs as ccrs
except ImportError:
    ccrs = None

# Import model registry
from model_registry import get_baseline_models
from normalization_utils import get_or_compute_stats

#%%
# ============ CONFIGURATION ============
# Model checkpoint path
model_name = "disco_epd_W9_MB1_D66_L_morlet"  # Name of your checkpoint directory
registry_name = "disco_epd_W9_MB1_D66_L_morlet"  # Architecture name in model_registry (set to None to use model_name)
checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints", model_name)
checkpoint_file = "checkpoint_stable1.pt"  # or "best_checkpoint.pt"

# Grid parameters (must match training)
nlat, nlon = 192, 288
grid = "equiangular"
in_chans, out_chans = 3, 3

# Rollout settings
max_rollout_steps = 500      # Maximum steps to run
blowup_threshold =50       # Stop when std grows by this factor
apply_spectral_filter = True  # Toggle this to compare filtered vs unfiltered
run_unfiltered = False        # Set to False to skip unfiltered rollout (faster)

# Initial condition source
# Option 1: From NC file
use_nc_file = True
nc_path = "/glade/derecho/scratch/idavis/TH_SWE_output/disco_epd_W9_MB1_D66_L_morlet_validation_steps28.nc"
ic_sample_idx = 0
ic_time_idx = 0  # Use the original IC (time=0)

# Option 2: Random IC from solver (if use_nc_file=False)
# Will generate a fresh random IC

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#%%
# ============ LOAD MODEL ============
# Use registry_name if specified, otherwise use model_name
arch_name = registry_name if registry_name else model_name

baseline_models = get_baseline_models(
    img_size=(nlat, nlon), 
    in_chans=in_chans, 
    out_chans=out_chans, 
    residual_prediction=True, 
    grid=grid
)

if arch_name not in baseline_models:
    print(f"Available models: {list(baseline_models.keys())}")
    raise ValueError(f"Architecture '{arch_name}' not found in registry")

model = baseline_models[arch_name]().to(device)
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded checkpoint: {checkpoint_path}")
else:
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

model.eval()
print(f"Model: {model_name}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

#%%
# ============ SETUP SPECTRAL FILTER ============
# Match the solver's spectral truncation
lmax = ceil(nlat / 3)  # = 64 for nlat=192
mmax = lmax

# Create SHT transforms for filtering
sht = RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid).to(device)
isht = InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid).to(device)

def spectral_filter(x):
    """Apply spectral truncation: grid -> spectral (lmax) -> grid.
    
    This removes all modes above lmax, exactly like the SWE solver does.
    Input: (batch, channels, lat, lon) or (channels, lat, lon)
    """
    squeeze = False
    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze = True
    
    # Apply SHT to each channel
    batch, chans, h, w = x.shape
    filtered = []
    for c in range(chans):
        spec = sht(x[:, c])
        filtered.append(isht(spec))
    
    result = torch.stack(filtered, dim=1)
    if squeeze:
        result = result.squeeze(0)
    return result

print(f"Spectral filter: lmax={lmax}, mmax={mmax}")
print(f"  Filtering removes modes l > {lmax} (wavelengths < {nlat/lmax:.1f} grid points)")

#%%
# ============ GET INITIAL CONDITION ============
# Create solver for normalization stats
dt_solver = 150
solver = ShallowWaterSolver(nlat, nlon, dt_solver, lmax=lmax, mmax=mmax, grid=grid)
solver = solver.to(device).float()

# Get normalization stats (cached, matching training)
mach = 0.2
dt = 600  # Output time step - matches training
norm_stats = get_or_compute_stats(
    solver=solver,
    cache_dir=checkpoint_dir,  # Use checkpoint dir for cache
    n_samples=1000,
    mach=mach,
    spinup_steps=50,
    dt=dt,
    dt_solver=dt_solver,
    show_progress=True
)


inp_mean = norm_stats["mean"]
inp_var = norm_stats["var"]

if use_nc_file:
    ds = xr.open_dataset(nc_path)
    ic_normalized = torch.tensor(
        ds['truth'][ic_sample_idx, ic_time_idx, :, :, :].values,
        dtype=torch.float32, device=device
    )
    ds.close()
    print(f"Loaded IC from NC file: sample {ic_sample_idx}, time {ic_time_idx}")
else:
    # Generate random IC
    ic_normalized = (random_ic_grid - inp_mean) / torch.sqrt(inp_var)
    print("Using random IC from solver")

print(f"IC shape: {ic_normalized.shape}")
print(f"IC range: [{ic_normalized.min():.3f}, {ic_normalized.max():.3f}]")

#%%
# ============ RUN ROLLOUT ============
# Run three parallel rollouts: ML filtered, ML unfiltered, and Solver ground truth

results_filtered = [ic_normalized.unsqueeze(0).clone()]
results_unfiltered = [ic_normalized.unsqueeze(0).clone()] if run_unfiltered else []
results_solver = [ic_normalized.clone()]  # Solver ground truth

current_filtered = ic_normalized.unsqueeze(0)  # Add batch dim
current_unfiltered = ic_normalized.unsqueeze(0) if run_unfiltered else None

# For solver: convert IC to physical units and spectral space
ic_physical = ic_normalized * torch.sqrt(inp_var) + inp_mean
current_solver_spec = solver.grid2spec(ic_physical)

# Solver parameters (match training)
dt = 600  # Output time step
solver_steps_per_output = dt // dt_solver  # = 4

print(f"\nRunning rollout (max {max_rollout_steps} steps, stop if {blowup_threshold}x blowup)...")
print(f"Spectral filtering: {'ENABLED' if apply_spectral_filter else 'DISABLED'}")
print(f"Unfiltered rollout: {'ENABLED' if run_unfiltered else 'DISABLED'}")
print(f"Also running solver for ground truth (dt={dt}s, {solver_steps_per_output} sub-steps)")

# Track initial std for blowup detection
initial_std = ic_normalized.std().item()
blowup_limit = initial_std * blowup_threshold
actual_steps = 0

with torch.no_grad():
    for step in range(max_rollout_steps):
        # Filtered ML rollout
        pred_filtered = model(current_filtered)
        if apply_spectral_filter:
            pred_filtered = spectral_filter(pred_filtered)
        results_filtered.append(pred_filtered.clone())
        current_filtered = pred_filtered
        
        # Unfiltered ML rollout (optional)
        if run_unfiltered:
            pred_unfiltered = model(current_unfiltered)
            results_unfiltered.append(pred_unfiltered.clone())
            current_unfiltered = pred_unfiltered
        
        # Solver ground truth
        current_solver_spec = solver.timestep(current_solver_spec, solver_steps_per_output)
        solver_grid = solver.spec2grid(current_solver_spec)
        solver_normalized = (solver_grid - inp_mean) / torch.sqrt(inp_var)
        results_solver.append(solver_normalized.clone())
        
        actual_steps = step + 1
        
        # Check for NaN explosion
        if not torch.isfinite(pred_filtered).all():
            print(f"  FILTERED rollout hit NaN at step {step+1}")
            break
        if run_unfiltered and not torch.isfinite(pred_unfiltered).all():
            print(f"  UNFILTERED rollout hit NaN at step {step+1}")
        
        # Check for blowup (filtered exceeds threshold)
        filt_std = pred_filtered.std().item()
        if filt_std > blowup_limit:
            print(f"  FILTERED blew up at step {step+1} (std={filt_std:.2f} > {blowup_limit:.2f})")
            break
        
        if (step + 1) % 10 == 0:
            if run_unfiltered:
                unfilt_std = pred_unfiltered.std().item() if torch.isfinite(pred_unfiltered).all() else float('nan')
            else:
                unfilt_std = float('nan')
            solver_std = solver_normalized.std().item()
            print(f"  Step {step+1}: filtered={filt_std:.3f}, unfiltered={unfilt_std:.3f}, solver={solver_std:.3f}")

n_rollout_steps = actual_steps  # For saving

# Stack results properly
results_filtered = torch.stack([r.squeeze(0) for r in results_filtered], dim=0)
if run_unfiltered:
    results_unfiltered_stacked = torch.stack([r.squeeze(0) for r in results_unfiltered], dim=0)
else:
    results_unfiltered_stacked = None
results_solver_stacked = torch.stack(results_solver, dim=0)

print(f"\nRollout complete!")
print(f"Results shape: {results_filtered.shape}")
print(f"Filtered - Final:   [{results_filtered[-1].min():.3f}, {results_filtered[-1].max():.3f}]")
if run_unfiltered:
    print(f"Unfiltered - Final: [{results_unfiltered_stacked[-1].min():.3f}, {results_unfiltered_stacked[-1].max():.3f}]")
print(f"Solver - Final:     [{results_solver_stacked[-1].min():.3f}, {results_solver_stacked[-1].max():.3f}]")

#%%
# ============ SAVE TO NETCDF ============
# Saves filtered, unfiltered, and solver ground truth

# Stack for saving: shape (1, n_steps+1, 3, nlat, nlon) - single "sample"
pred_filtered_np = results_filtered.cpu().numpy()[np.newaxis, ...]
if run_unfiltered:
    pred_unfiltered_np = results_unfiltered_stacked.cpu().numpy()[np.newaxis, ...]
else:
    pred_unfiltered_np = np.full_like(pred_filtered_np, np.nan)  # Placeholder
solver_truth_np = results_solver_stacked.cpu().numpy()[np.newaxis, ...]

# Define coords
times = np.arange(n_rollout_steps + 1)
samples = np.arange(1)  # Single sample
chans = np.arange(3)
lats = np.linspace(-90, 90, nlat)
lons = np.linspace(0, 360, nlon, endpoint=False)

ds_out = xr.Dataset(
    data_vars={
        "filtered": (("sample", "time", "channel", "lat", "lon"), pred_filtered_np),
        "unfiltered": (("sample", "time", "channel", "lat", "lon"), pred_unfiltered_np),
        "truth": (("sample", "time", "channel", "lat", "lon"), solver_truth_np),
        # Also save as "prediction" for compatibility with plot_swe_results.py
        "prediction": (("sample", "time", "channel", "lat", "lon"), pred_filtered_np),
    },
    coords={
        "sample": samples,
        "time": times,
        "channel": chans,
        "lat": lats,
        "lon": lons,
    },
    attrs={
        "model_name": model_name,
        "spectral_filter_applied": str(apply_spectral_filter),
        "lmax": lmax,
        "n_rollout_steps": n_rollout_steps,
        "note": "filtered=ML+filter, unfiltered=ML raw, truth=solver, prediction=filtered (for plot_swe_results.py compat)"
    }
)

# Save to same location as TH_SWE.py
scratch_dir = f"/glade/derecho/scratch/{os.environ['USER']}/TH_SWE_output"
if not os.path.exists(scratch_dir):
    os.makedirs(scratch_dir, exist_ok=True)

filter_tag = "filtered" if apply_spectral_filter else "unfiltered"
save_path = os.path.join(scratch_dir, f"{model_name}_{filter_tag}_rollout_steps{n_rollout_steps}.nc")
print(f"\nSaving NetCDF to {save_path}...")
ds_out.to_netcdf(save_path)
print(f"Save complete!")
print(f"  prediction = filtered rollout")
print(f"  truth = unfiltered rollout (for comparison)")


#%%
# ============ PLOT: COMPARE FILTERED VS UNFILTERED ============
channel_idx = 2  # 0=h, 1=u, 2=v
time_idx = min(n_rollout_steps, 20)  # Step to visualize
lat_cutoff = 85

channel_names = ['h (height)', 'u (velocity)', 'v (velocity)']
lat = np.linspace(90, -90, nlat)
lon = np.linspace(0, 360, nlon, endpoint=False)
lat_mask = np.abs(lat) <= lat_cutoff

# Get data at selected timestep
ic_data = results_filtered[0, channel_idx].cpu().numpy()[lat_mask, :]

# Filtered
if results_filtered.shape[0] > time_idx:
    filt_data = results_filtered[time_idx, channel_idx].cpu().numpy()[lat_mask, :]
else:
    filt_data = results_filtered[-1, channel_idx].cpu().numpy()[lat_mask, :]

# Unfiltered (from the stored list)
try:
    unfilt_tensor = results_unfiltered[time_idx].squeeze(0)
    if torch.isfinite(unfilt_tensor).all():
        unfilt_data = unfilt_tensor[channel_idx].cpu().numpy()[lat_mask, :]
    else:
        unfilt_data = np.full_like(filt_data, np.nan)
except:
    unfilt_data = np.full_like(filt_data, np.nan)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Use same color scale
valid_data = [d for d in [ic_data, filt_data, unfilt_data] if np.isfinite(d).any()]
vmin = min(d.min() for d in valid_data)
vmax = max(d.max() for d in valid_data)

im0 = axes[0].pcolormesh(lon, lat[lat_mask], ic_data, cmap="viridis", vmin=vmin, vmax=vmax)
axes[0].set_title("t=0 (IC)")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].pcolormesh(lon, lat[lat_mask], filt_data, cmap="viridis", vmin=vmin, vmax=vmax)
axes[1].set_title(f"t={time_idx} FILTERED")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].pcolormesh(lon, lat[lat_mask], unfilt_data, cmap="viridis", vmin=vmin, vmax=vmax)
axes[2].set_title(f"t={time_idx} UNFILTERED")
plt.colorbar(im2, ax=axes[2])

plt.suptitle(f"{channel_names[channel_idx]} - Filtered vs Unfiltered Rollout")
plt.tight_layout()
plt.show()

#%%
# ============ PLOT: TIME SERIES OF STD (STABILITY CHECK) ============
filtered_stds = [results_filtered[t].std().item() for t in range(results_filtered.shape[0])]
unfiltered_stds = []
for r in results_unfiltered:
    if torch.isfinite(r).all():
        unfiltered_stds.append(r.std().item())
    else:
        unfiltered_stds.append(np.nan)

plt.figure(figsize=(10, 5))
plt.plot(filtered_stds, 'b-', label='Filtered', linewidth=2)
plt.plot(unfiltered_stds, 'r--', label='Unfiltered', linewidth=2)
plt.xlabel('Rollout Step')
plt.ylabel('Field Std')
plt.title('Rollout Stability: Filtered vs Unfiltered')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print summary
print(f"\n{'='*50}")
print(f"STABILITY SUMMARY")
print(f"{'='*50}")
print(f"Initial std: {filtered_stds[0]:.3f}")
print(f"Final std (filtered): {filtered_stds[-1]:.3f}")
print(f"Final std (unfiltered): {unfiltered_stds[-1]:.3f}")
print(f"Filtered growth: {filtered_stds[-1]/filtered_stds[0]:.2f}x")
if np.isfinite(unfiltered_stds[-1]):
    print(f"Unfiltered growth: {unfiltered_stds[-1]/unfiltered_stds[0]:.2f}x")
else:
    print(f"Unfiltered: EXPLODED")

#%%
