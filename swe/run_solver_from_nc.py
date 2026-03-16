#%%
# Run SWE solver using initial conditions from an existing NetCDF file
# Load a timestep from a validation NC file, solve for n steps, and plot results

import numpy as np
import os
import torch
import xarray as xr
import matplotlib.pyplot as plt
from math import ceil
#%%
from torch_harmonics.examples.shallow_water_equations import ShallowWaterSolver
from normalization_utils import get_or_compute_stats

try:
    import cartopy.crs as ccrs
    import cartopy.feature
except ImportError:
    print("Warning: cartopy not installed, spherical plots will fail.")
    ccrs = None

#%%
# ============ CONFIGURATION ============
# Path to your validation output NetCDF file
nc_path = "/glade/derecho/scratch/idavis/TH_SWE_output/disco_epd_W9_MB1_D66_L_morlet_validation_steps28.nc"
nc_path = "/glade/derecho/scratch/idavis/TH_SWE_output/disco_epd_W9_MB1_D66_L_morlet_filtered_rollout_steps500.nc"
nc_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_filtered_validation_steps92.nc"

# ---- Initial Condition Selection ----
# Select which sample and timestep from the NC file to use as IC for the solver
# ic_sample_idx: which random IC trajectory (0 to n_samples-1)
# ic_time_idx: which timestep along that trajectory (0 = original IC, higher = later in rollout)
# ic_source: "prediction" (ML model output) or "truth" (solver ground truth)
ic_sample_idx = 0
ic_time_idx = 0   # <-- Change this to pick any timestep from the NC file as your IC
ic_source = "prediction"  # <-- "prediction" or "truth"

# Number of solver steps to run FROM the selected IC
n_solver_steps = 24

# Solver parameters (should match what was used in training)
dt = 600          # Output time step in seconds
dt_solver = 150   # Internal solver time step
grid = "equiangular"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#%%
# ============ LOAD DATA AND RUN SOLVER ============
# Load the NC file
ds = xr.open_dataset(nc_path)
print(f"Loaded: {nc_path}")
print(f"  Available: {ds.dims['sample']} samples, {ds.dims['time']} timesteps per sample")

# Validate selection
if ic_sample_idx >= ds.dims['sample']:
    raise ValueError(f"ic_sample_idx={ic_sample_idx} out of bounds (max: {ds.dims['sample']-1})")
if ic_time_idx >= ds.dims['time']:
    raise ValueError(f"ic_time_idx={ic_time_idx} out of bounds (max: {ds.dims['time']-1})")

print(f"  Using {ic_source} from sample {ic_sample_idx}, timestep {ic_time_idx} as initial condition")

# Extract IC from NC file (shape: channel, lat, lon)
ic_normalized = torch.tensor(
    ds[ic_source][ic_sample_idx, ic_time_idx, :, :, :].values, 
    dtype=torch.float32
)
nlat, nlon = ic_normalized.shape[1], ic_normalized.shape[2]
print(f"  IC shape: {ic_normalized.shape}, grid: {nlat}x{nlon}")

#%%
# ============ DEBUG: VERIFY IC LOADED CORRECTLY ============
# Compare what we loaded to what plot_swe_results.py would show
debug_channel = 0  # Match what you're viewing in plot_swe_results.py

# Direct from xarray (same as plot_swe_results.py accesses it)
direct_from_ds = ds[ic_source][ic_sample_idx, ic_time_idx, debug_channel, :, :].values

# What we loaded into ic_normalized
from_tensor = ic_normalized[debug_channel, :, :].numpy()

print(f"Direct from DS - min: {direct_from_ds.min():.4f}, max: {direct_from_ds.max():.4f}")
print(f"From tensor    - min: {from_tensor.min():.4f}, max: {from_tensor.max():.4f}")
print(f"Arrays equal: {np.allclose(direct_from_ds, from_tensor)}")

# Quick side-by-side plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
im0 = axes[0].imshow(direct_from_ds, cmap='RdBu_r', origin='lower')
axes[0].set_title(f"Direct from NC (sample {ic_sample_idx}, time {ic_time_idx}, ch {debug_channel})")
plt.colorbar(im0, ax=axes[0])
im1 = axes[1].imshow(from_tensor, cmap='RdBu_r', origin='lower')
axes[1].set_title("Loaded into ic_normalized")
plt.colorbar(im1, ax=axes[1])
plt.tight_layout()
plt.show()

# Create solver
lmax = ceil(nlat / 3)
mmax = lmax
solver = ShallowWaterSolver(nlat, nlon, dt_solver, lmax=lmax, mmax=mmax, grid=grid)
solver = solver.to(device).float()

# Get normalization stats (cached, matching training)
mach = 0.2
checkpoint_dir = os.path.dirname(nc_path)  # Use output dir for cache, or specify your training checkpoints dir
norm_stats = get_or_compute_stats(
    solver=solver,
    cache_dir=checkpoint_dir,
    n_samples=1000,
    mach=mach,
    spinup_steps=100,
    dt=dt,
    dt_solver=dt_solver,
    show_progress=True
)
inp_mean = norm_stats["mean"]
inp_var = norm_stats["var"]

# Denormalize the IC (NC file has normalized data)
ic_grid = ic_normalized.to(device) * torch.sqrt(inp_var) + inp_mean

# Convert to spectral space
ic_spec = solver.grid2spec(ic_grid)

# Run solver
solver_steps_per_output = dt // dt_solver
results = [ic_normalized.to(device)]
current_spec = ic_spec.clone()

print(f"Running solver for {n_solver_steps} steps...")
for step in range(n_solver_steps):
    current_spec = solver.timestep(current_spec, solver_steps_per_output)
    current_grid = solver.spec2grid(current_spec)
    current_normalized = (current_grid - inp_mean) / torch.sqrt(inp_var)
    results.append(current_normalized)

results = torch.stack(results, dim=0)  # (n_steps+1, 3, nlat, nlon)
print(f"Done. Results shape: {results.shape}")

#%%
# ============ RECTILINEAR PLOT ============
# Configuration
time_idx = 8      # Which timestep to plot (0 to n_solver_steps)
channel_idx = 0    # 0: h (height), 1: u (velocity), 2: v (velocity)
lat_cutoff = 90

channel_names = ['h (height)', 'u (velocity)', 'v (velocity)']
var_name = channel_names[channel_idx]

# Create coordinate arrays
lat = np.linspace(90, -90, nlat)
lon = np.linspace(0, 360, nlon, endpoint=False)
lat_mask = np.abs(lat) <= lat_cutoff
lat_cropped = lat[lat_mask]

# Extract and crop data
data_t0 = results[0, channel_idx, :, :].cpu().numpy()[lat_mask, :]
data_t = results[time_idx, channel_idx, :, :].cpu().numpy()[lat_mask, :]
diff = data_t - data_t0

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

vmin = min(data_t0.min(), data_t.min())
vmax = max(data_t0.max(), data_t.max())

im0 = axes[0].pcolormesh(lon, lat_cropped, data_t0, cmap="RdBu_r", vmin=vmin, vmax=vmax)
axes[0].set_title(f"t=0 (IC)")
axes[0].invert_yaxis()
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].pcolormesh(lon, lat_cropped, data_t, cmap="RdBu_r", vmin=vmin, vmax=vmax)
axes[1].set_title(f"t={time_idx}")
axes[1].invert_yaxis()
plt.colorbar(im1, ax=axes[1])

diff_max = np.abs(diff).max()
im2 = axes[2].pcolormesh(lon, lat_cropped, diff, cmap="RdBu", vmin=-diff_max, vmax=diff_max)
axes[2].set_title(f"Difference (t={time_idx} - t=0)")
axes[2].invert_yaxis()
plt.colorbar(im2, ax=axes[2])

plt.suptitle(f"SWE Solver Results: {var_name} | IC from sample {ic_sample_idx}, time {ic_time_idx}")
plt.tight_layout()
plt.show()

#%%
# ============ SPHERICAL PLOT ============
# Configuration
time_idx = 0      # Which timestep to plot
channel_idx = 0    # 0: h, 1: u, 2: v
lat_cutoff = 90
central_lat = 0
central_lon = 0

channel_names = ['h (height)', 'u (velocity)', 'v (velocity)']
var_name = channel_names[channel_idx]

def get_projection(proj_type, clat=0, clon=0):
    if proj_type == "orthographic":
        return ccrs.Orthographic(central_latitude=clat, central_longitude=clon)
    elif proj_type == "robinson":
        return ccrs.Robinson(central_longitude=clon)
    elif proj_type == "mollweide":
        return ccrs.Mollweide(central_longitude=clon)
    return ccrs.PlateCarree()

def plot_sphere_data(data, ax, cmap="RdBu", vmin=None, vmax=None):
    lon = np.linspace(0, 360, data.shape[1], endpoint=False)
    lat = np.linspace(90, -90, data.shape[0])
    Lon, Lat = np.meshgrid(lon, lat)
    im = ax.pcolormesh(Lon, Lat, data, cmap=cmap, transform=ccrs.PlateCarree(),
                       antialiased=False, vmin=vmin, vmax=vmax)
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1,
                 color="gray", alpha=0.6, linestyle="--")
    return im

# Apply latitude mask - broadcast to full 2D shape for boolean indexing
lat = np.linspace(90, -90, nlat)
lat_mask_2d = np.broadcast_to(np.abs(lat[:, None]) > lat_cutoff, (nlat, nlon))

data_t0 = results[0, channel_idx, :, :].cpu().numpy().copy()
data_t = results[time_idx, channel_idx, :, :].cpu().numpy().copy()
data_t0[lat_mask_2d] = np.nan
data_t[lat_mask_2d] = np.nan
diff = data_t - data_t0

vmin = np.nanmin([data_t0, data_t])
vmax = np.nanmax([data_t0, data_t])
diff_max = np.nanmax(np.abs(diff))

fig = plt.figure(figsize=(18, 6))

proj = get_projection("orthographic", central_lat, central_lon)

ax1 = fig.add_subplot(1, 3, 1, projection=proj)
plot_sphere_data(data_t0, ax1, vmin=vmin, vmax=vmax)
ax1.set_title(f"t=0 (IC)")

ax2 = fig.add_subplot(1, 3, 2, projection=proj)
plot_sphere_data(data_t, ax2, vmin=vmin, vmax=vmax)
ax2.set_title(f"t={time_idx}")

ax3 = fig.add_subplot(1, 3, 3, projection=proj)
plot_sphere_data(diff, ax3, cmap="RdBu_r", vmin=-diff_max, vmax=diff_max)
ax3.set_title(f"Difference")

plt.suptitle(f"Spherical View: {var_name} | IC from sample {ic_sample_idx}, time {ic_time_idx}")
plt.tight_layout()
plt.show()

#%%
