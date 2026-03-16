#%%
# Sandbox for 
# experimenting with high-frequency noise injection
# Goal: Create initial conditions with similar wave patterns to what the ML model produces

import numpy as np
import torch
import xarray as xr
import matplotlib.pyplot as plt
import math
from math import ceil
from torch_harmonics.examples.shallow_water_equations import ShallowWaterSolver

try:
    import cartopy.crs as ccrs
except ImportError:
    ccrs = None

def solve_balance_eqn(solver, uspec):
    """
    Solve for Phi such that d(div)/dt = 0 at t=0 (Nonlinear Balance Equation).
    Del^2(Phi + K) = div( (zeta+f) * u_rot )
    """
    # 1. Get u, v from spec
    uvgrid = solver.getuv(uspec[1:]) # (2, nlat, nlon)
    u, v = uvgrid[0], uvgrid[1]
    
    # 2. Get Vorticity (zeta)
    vrtdivgrid = solver.spec2grid(uspec[1:])
    zeta = vrtdivgrid[0]
    
    # 3. Absolute Vorticity (eta)
    f = solver.coriolis.squeeze().view(-1, 1) # matches (nlat, 1) or (nlat, nlon)
    eta = zeta + f
    
    # 4. A = eta * u_vec
    A = uvgrid * eta.unsqueeze(0)
    
    # 5. Curl(A)
    # solver.vrtdivspec returns (vort, div) of vector
    # vort(A) = curl(A)
    Aspec = solver.vrtdivspec(A)
    curl_A = Aspec[0]
    
    # 6. Kinetic Energy K
    K = 0.5 * (u**2 + v**2)
    Kspec = solver.grid2spec(K)
    
    # 7. Solve for Phi
    # Del^2 Phi = curl(A) - Del^2 K
    # Phi = invlap[ curl(A) ] - K
    Phi = solver.invlap * curl_A - Kspec
    
    # 8. Add mean height
    Phi[0, 0] += solver.gravity * solver.havg * math.sqrt(4 * math.pi)
    
    return Phi

def get_geostrophic_divergence(solver, uspec):
    """
    Compute analytic geostrophic divergence: D_g = - (beta * v) / f.
    Tapers to 0 at the equator to avoid singularity.
    Quantifies the divergence required to maintain geostrophic balance on a varying f-plane (sphere).
    """
    uvgrid = solver.getuv(uspec[1:])
    v = uvgrid[1]
    
    lats = solver.lats.squeeze()
    
    # Gradient of f (beta term)
    # beta = df/dy = (1/R) * df/dlat = (1/R) * 2*Omega*cos(lat)
    # But term is beta/f = (2Om cos) / (2Om sin) * (1/R) = cot(lat) / R
    
    # Window function to kill divergence at equator where formula blows up
    # w(lat) = 1 - exp(-(lat/width)^2)
    # width ~ 10-20 degrees
    w = 1.0 - torch.exp(-(lats / (15 * math.pi/180))**2)
    
    # Broadcast lats for grid operations
    lats_2d = lats.view(-1, 1)
    w_2d = w.view(-1, 1)
    
    # D_g = -(v/R) * cot(lat)
    # Use 1/tan(lat)
    div_grid = - (v / solver.radius) * (1.0 / torch.tan(lats_2d + 1e-9))
    div_grid = div_grid * w_2d
    
    # Convert to spectral
    div_spec = solver.grid2spec(div_grid)
    return div_spec

def balanced_random_initial_condition(solver, mach=0.1, llimit=25, div_mach=0.0):
    """
    Generate a random initial condition in geostrophic balance.
    
    Process:
    1. Generate random Vorticity (scaled by mach).
    2. Solve Nonlinear Balance Eq for Height.
    3. Compute Analytic Geostrophic Divergence.
    4. (Optional) Add random divergence noise (gravity waves) if div_mach > 0.
    """
    device = solver.lap.device
    ctype = torch.complex128 if solver.lap.dtype == torch.float64 else torch.complex64
    
    uspec = torch.zeros(3, solver.lmax, solver.mmax, dtype=ctype, device=device)
    
    # --- 1. Random Vorticity ---
    # Scale matches standard random_initial_condition
    scale_vort = mach * torch.sqrt(solver.gravity * solver.havg) / solver.radius
    
    # Restrict to llimit
    lmin = 0 # Assume 0 for base
    uspec[1, lmin:llimit, lmin:llimit] = (
        torch.sqrt(torch.tensor(4 * torch.pi / llimit / (llimit + 1), device=device, dtype=ctype)) 
        * torch.randn_like(uspec[1, lmin:llimit, lmin:llimit])
    ) * scale_vort
    
    # --- 2. Balanced Height ---
    # Solve Phi from Vorticity (assuming Div=0 for the balance inversion steps)
    phispec = solve_balance_eqn(solver, uspec)
    uspec[0] = phispec
    
    # --- 3. Geostrophic Divergence ---
    # Compute D_g consistent with the rotational wind
    div_geo = get_geostrophic_divergence(solver, uspec)
    uspec[2] = div_geo
    
    # --- 4. Optional Random Divergence (Gravity Waves) ---
    if div_mach > 0:
        scale_div = div_mach * torch.sqrt(solver.gravity * solver.havg) / solver.radius
        noise_div = torch.zeros_like(uspec[2])
        noise_div[lmin:llimit, lmin:llimit] = (
            torch.sqrt(torch.tensor(4 * torch.pi / llimit / (llimit + 1), device=device, dtype=ctype)) 
            * torch.randn_like(noise_div[lmin:llimit, lmin:llimit])
        ) * scale_div
        uspec[2] += noise_div
        
    return uspec

def gaussian_bump_initial_condition(solver):
    """
    Generate a static 'lifted height' initial condition.
    Gaussian bump in geopotential height centered at 45N, 180E.
    Zero vorticity and zero divergence.
    """
    device = solver.lap.device
    ctype = torch.complex128 if solver.lap.dtype == torch.float64 else torch.complex64
    
    # Grid coordinates
    # lats: (nlat,), lons: (nlon,)
    lats = solver.lats.squeeze()
    lons = solver.lons.squeeze()
    Lats, Lons = torch.meshgrid(lats, lons, indexing='ij')
    
    # Bump parameters
    lat0 = 45.0 * math.pi / 180.0
    lon0 = math.pi # 180 degrees
    width = 20.0 * math.pi / 180.0
    amp = solver.hamp * 5.0 # Reduced amplitude (was 10.0)
    
    # Distance squared (Great circle distance approximation for Gaussian)
    # dist = arccos(sin(lat)*sin(lat0) + cos(lat)*cos(lat0)*cos(lon-lon0))
    # Use a finite radius for the Cosine Bell to ensure exact zero support
    radius_bump = 40.0 * math.pi / 180.0 
    amp = solver.hamp * 2.0  # Reduced amplitude as requested
    
    # Distance (Great circle central angle)
    central_angle = torch.acos(
        torch.clamp(
            torch.sin(Lats) * math.sin(lat0) + 
            torch.cos(Lats) * math.cos(lat0) * torch.cos(Lons - lon0), 
            -1.0, 1.0
        )
    )
    dist = central_angle
    
    # Cosine Bell: 0.5 * A * (1 + cos(pi * r / R)) for r < R
    h_perturbation = torch.zeros_like(dist)
    mask = dist < radius_bump
    h_perturbation[mask] = 0.5 * amp * (1.0 + torch.cos(math.pi * dist[mask] / radius_bump))
    
    h_grid = solver.havg + h_perturbation
    
    # Scale by gravity (solver expects geopotential)
    phi_grid = solver.gravity * h_grid
    
    # Convert to spectral
    phi_spec = solver.grid2spec(phi_grid)
            
    # Full state: [Phi, Vort, Div]
    uspec = torch.zeros(3, solver.lmax, solver.mmax, dtype=ctype, device=device)
    uspec[0] = phi_spec
    # Vorticity (1) and Divergence (2) remain zero
    
    return uspec

# Legacy wrapper for compatibility with existing sandbox code calls (renaming for clarity)
def random_initial_condition_configurable(solver, mach=0.1, llimit=25, lmin=0, balanced=False, ic_type="random", div_mach=0.0):
    """
    Configurable IC generation wrapper.
    Args:
        ic_type: "random", "balanced_random", or "gaussian_bump"
        balanced: (Legacy) if True, overrides ic_type to "balanced_random"
    """
    if balanced:
        ic_type = "balanced_random"
        
    if ic_type == "gaussian_bump":
        return gaussian_bump_initial_condition(solver)
    elif ic_type == "balanced_random":
        return balanced_random_initial_condition(solver, mach=mach, llimit=llimit, div_mach=div_mach)
    
    # Original Unbalanced Logic ("random")
    device = solver.lap.device
    ctype = torch.complex128 if solver.lap.dtype == torch.float64 else torch.complex64
    
    # Clamp to valid range
    lmin = max(0, min(lmin, solver.lmax - 1))
    llimit = min(llimit, solver.lmax)
    mlimit = min(llimit, solver.mmax)
    mmin = min(lmin, mlimit)
    
    # Number of active modes for normalization
    n_active = max(1, llimit - lmin)
    
    # Initialize spectral coefficients
    uspec = torch.zeros(3, solver.lmax, solver.mmax, dtype=ctype, device=device)
    
    # Add random energy to modes in range [lmin, llimit)
    uspec[:, lmin:llimit, mmin:mlimit] = (
        torch.sqrt(torch.tensor(4 * torch.pi / n_active / (n_active + 1), device=device, dtype=ctype)) 
        * torch.randn_like(uspec[:, lmin:llimit, mmin:mlimit])
    )
    
    # Scale height field
    uspec[0] = solver.gravity * solver.hamp * uspec[0]
    # Add mean height (always at mode 0,0 regardless of lmin)
    uspec[0, 0, 0] += torch.sqrt(torch.tensor(4 * torch.pi, device=device, dtype=ctype)) * solver.havg * solver.gravity
    # Scale vorticity/divergence by mach number
    uspec[1:] = mach * uspec[1:] * torch.sqrt(solver.gravity * solver.havg) / solver.radius
    
    return torch.tril(uspec)

# ============ SETUP ============
nlat, nlon = 192, 288
dt_solver = 150
grid = "equiangular"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create solver (needed for generating base ICs)
lmax = ceil(nlat / 3)
mmax = lmax
solver = ShallowWaterSolver(nlat, nlon, dt_solver, lmax=lmax, mmax=mmax, grid=grid)
solver = solver.to(device).float()

# Generate a clean random IC as base using configurable parameters
torch.manual_seed(42)
# ============ IC GENERATION PARAMETERS ============
ic_mach = 0.4
ic_llimit = 2
ic_lmin = 0
ic_balanced = False  # Legacy flag
ic_type = "random" # Options: "random", "balanced_random", "gaussian_bump"

print(f"IC Parameters: type={ic_type}, mach={ic_mach}, lmin={ic_lmin}, llimit={ic_llimit}")
base_ic_spec = random_initial_condition_configurable(solver, mach=ic_mach, llimit=ic_llimit, lmin=ic_lmin, balanced=ic_balanced, ic_type=ic_type)

base_ic_grid = solver.spec2grid(base_ic_spec)

# Compute normalization stats
inp_mean = torch.mean(base_ic_grid, dim=(-1, -2)).reshape(-1, 1, 1)
inp_var = torch.var(base_ic_grid, dim=(-1, -2)).reshape(-1, 1, 1)
# Handle standard deviation of 0 (constant fields like U=0, V=0)
inp_var = torch.where(inp_var < 1e-10, torch.ones_like(inp_var), inp_var)

base_ic_normalized = (base_ic_grid - inp_mean) / torch.sqrt(inp_var)

print(f"Base IC shape: {base_ic_normalized.shape}")
print(f"Base IC range: [{base_ic_normalized.min():.3f}, {base_ic_normalized.max():.3f}]")

#%%
# ============ SANITY CHECK: VERIFY SOLVER STABILITY ============
# Run solver DIRECTLY without any normalization pipeline to verify it's stable
n_test_steps = 100
test_dt = 600  # Same as training - 600s per output step
test_sub_steps = test_dt // dt_solver  # = 4 sub-steps

print(f"Sanity check: Running solver for {n_test_steps} steps (dt={test_dt}s, {test_sub_steps} sub-steps each)")
print(f"Total simulation time: {n_test_steps * test_dt / 3600:.1f} hours")

# Use the ORIGINAL spectral IC directly (no grid roundtrip)
torch.manual_seed(42)
test_spec = random_initial_condition_configurable(solver, mach=ic_mach, llimit=ic_llimit, lmin=ic_lmin, balanced=ic_balanced, ic_type=ic_type)
test_grid_0 = solver.spec2grid(test_spec)  # Initial state

current_spec = test_spec.clone()
for i in range(n_test_steps):
    current_spec = solver.timestep(current_spec, test_sub_steps)
    
test_grid_final = solver.spec2grid(current_spec)

print(f"Initial: min={test_grid_0.min():.2f}, max={test_grid_0.max():.2f}")
print(f"Final:   min={test_grid_final.min():.2f}, max={test_grid_final.max():.2f}")
print(f"Ratio:   {test_grid_final.abs().max() / test_grid_0.abs().max():.2f}x")

# If this shows >2x growth, the solver itself is unstable
# If this is stable (~1x), then our normalization pipeline has a bug

#%%
# ============ NOISE INJECTION METHODS ============

def add_checkerboard_noise(field, amplitude=0.1):
    """Add pure checkerboard (Nyquist) noise - alternating +/- at each grid point."""
    nlat, nlon = field.shape[-2], field.shape[-1]
    checker = np.ones((nlat, nlon))
    checker[::2, ::2] = -1
    checker[1::2, 1::2] = -1
    checker = torch.tensor(checker, dtype=field.dtype, device=field.device)
    return field + amplitude * checker

def add_diagonal_stripes(field, amplitude=0.1, wavelength=4):
    """Add diagonal stripe pattern (similar to the ML artifact pattern)."""
    nlat, nlon = field.shape[-2], field.shape[-1]
    lat_idx = np.arange(nlat)[:, None]
    lon_idx = np.arange(nlon)[None, :]
    # Diagonal pattern
    pattern = np.sin(2 * np.pi * (lat_idx + lon_idx) / wavelength)
    pattern = torch.tensor(pattern, dtype=field.dtype, device=field.device)
    return field + amplitude * pattern

def add_high_freq_spectral_noise(field, solver, amplitude=0.1, l_min=50):
    """Add noise only to high spherical harmonic modes (l > l_min)."""
    # Convert to spectral space
    spec = solver.grid2spec(field)
    # Add noise to high-l modes
    noise = amplitude * torch.randn_like(spec)
    noise[:, :l_min, :] = 0  # Zero out low modes
    spec = spec + noise
    return solver.spec2grid(spec)

def add_gridscale_random_noise(field, amplitude=0.1):
    """Add uniform random noise at every grid point."""
    noise = amplitude * torch.randn_like(field)
    return field + noise

def add_latitude_varying_noise(field, amplitude=0.1, wavelength=4):
    """Add noise with amplitude varying by latitude (strongest at equator)."""
    nlat, nlon = field.shape[-2], field.shape[-1]
    lat = np.linspace(90, -90, nlat)
    lat_weight = np.cos(np.radians(lat))[:, None]  # Max at equator
    
    lat_idx = np.arange(nlat)[:, None]
    lon_idx = np.arange(nlon)[None, :]
    pattern = np.sin(2 * np.pi * (lat_idx + lon_idx) / wavelength)
    pattern = pattern * lat_weight
    pattern = torch.tensor(pattern, dtype=field.dtype, device=field.device)
    return field + amplitude * pattern

#%%
# ============ EXPERIMENT: TRY DIFFERENT NOISE TYPES ============
# Configuration - EDIT THESE to experiment
channel_idx = 2      # Which channel to visualize (0=h, 1=u, 2=v)
noise_type = "gridscale_random"  # Options below:
# "checkerboard"       - Pure Nyquist checkerboard
# "diagonal_stripes"   - Diagonal stripe pattern (wavelength=4)
# "high_freq_spectral" - Spectral noise in high-l modes
# "gridscale_random"   - Random noise at every point  
# "latitude_varying"   - Stripes weighted by latitude

noise_amplitude = 0  # Adjust this to control noise strength
stripe_wavelength = 4   # For stripe-based patterns (smaller = finer)
l_min_spectral = 1     # For spectral noise: modes above this get noise

# Apply noise
clean = base_ic_normalized.clone()

if noise_type == "checkerboard":
    noisy = add_checkerboard_noise(clean, amplitude=noise_amplitude)
elif noise_type == "diagonal_stripes":
    noisy = add_diagonal_stripes(clean, amplitude=noise_amplitude, wavelength=stripe_wavelength)
elif noise_type == "high_freq_spectral":
    noisy = add_high_freq_spectral_noise(clean, solver, amplitude=noise_amplitude, l_min=l_min_spectral)
elif noise_type == "gridscale_random":
    noisy = add_gridscale_random_noise(clean, amplitude=noise_amplitude)
elif noise_type == "latitude_varying":
    noisy = add_latitude_varying_noise(clean, amplitude=noise_amplitude, wavelength=stripe_wavelength)
else:
    noisy = clean.clone()
    print(f"Unknown noise_type: {noise_type}")

noise_only = noisy - clean

print(f"Noise type: {noise_type}")
print(f"Clean range: [{clean[channel_idx].min():.3f}, {clean[channel_idx].max():.3f}]")
print(f"Noisy range: [{noisy[channel_idx].min():.3f}, {noisy[channel_idx].max():.3f}]")
print(f"Noise range: [{noise_only[channel_idx].min():.3f}, {noise_only[channel_idx].max():.3f}]")

#%%
# ============ PLOT: RECTILINEAR VIEW ============
channel_names = ['h (height)', 'u (velocity)', 'v (velocity)']
lat_cutoff = 90

lat = np.linspace(90, -90, nlat)
lon = np.linspace(0, 360, nlon, endpoint=False)
lat_mask = np.abs(lat) <= lat_cutoff

clean_plot = clean[channel_idx].cpu().numpy()[lat_mask, :]
noisy_plot = noisy[channel_idx].cpu().numpy()[lat_mask, :]
noise_plot = noise_only[channel_idx].cpu().numpy()[lat_mask, :]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

vmin = min(clean_plot.min(), noisy_plot.min())
vmax = max(clean_plot.max(), noisy_plot.max())

im0 = axes[0].pcolormesh(lon, lat[lat_mask], clean_plot, cmap="viridis", vmin=vmin, vmax=vmax)
axes[0].set_title("Clean IC")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].pcolormesh(lon, lat[lat_mask], noisy_plot, cmap="viridis", vmin=vmin, vmax=vmax)
axes[1].set_title(f"With {noise_type} noise (amp={noise_amplitude})")
plt.colorbar(im1, ax=axes[1])

noise_max = np.abs(noise_plot).max()
im2 = axes[2].pcolormesh(lon, lat[lat_mask], noise_plot, cmap="RdBu_r", vmin=-noise_max, vmax=noise_max)
axes[2].set_title("Noise only")
plt.colorbar(im2, ax=axes[2])

plt.suptitle(f"{channel_names[channel_idx]} - {noise_type}")
plt.tight_layout()
plt.show()

#%%
# ============ PLOT: SPHERICAL VIEW (OPTIONAL) ============
if ccrs is not None:
    central_lat, central_lon = 30, 0
    
    def plot_sphere(data, ax, cmap="viridis", vmin=None, vmax=None):
        lon = np.linspace(0, 360, data.shape[1], endpoint=False)
        lat = np.linspace(90, -90, data.shape[0])
        Lon, Lat = np.meshgrid(lon, lat)
        im = ax.pcolormesh(Lon, Lat, data, cmap=cmap, transform=ccrs.PlateCarree(),
                           antialiased=False, vmin=vmin, vmax=vmax)
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1,
                     color="gray", alpha=0.6, linestyle="--")
        return im
    
    # Mask poles
    lat_mask_2d = np.broadcast_to(np.abs(lat[:, None]) > lat_cutoff, (nlat, nlon))
    
    clean_sphere = clean[channel_idx].cpu().numpy().copy()
    noisy_sphere = noisy[channel_idx].cpu().numpy().copy()
    clean_sphere[lat_mask_2d] = np.nan
    noisy_sphere[lat_mask_2d] = np.nan
    
    fig = plt.figure(figsize=(14, 6))
    proj = ccrs.Orthographic(central_latitude=central_lat, central_longitude=central_lon)
    
    ax1 = fig.add_subplot(1, 2, 1, projection=proj)
    plot_sphere(clean_sphere, ax1, vmin=vmin, vmax=vmax)
    ax1.set_title("Clean IC")
    
    ax2 = fig.add_subplot(1, 2, 2, projection=proj)
    plot_sphere(noisy_sphere, ax2, vmin=vmin, vmax=vmax)
    ax2.set_title(f"With {noise_type} noise")
    
    plt.suptitle(f"Spherical view: {channel_names[channel_idx]}")
    plt.tight_layout()
    plt.show()

#%%
# ============ RUN SOLVER: TEST STABILITY WITH NOISY IC ============
# Configuration
n_solver_steps = 500      # Number of steps to integrate
dt = 600                 # Output time step (same as training!)



# Re-compute noisy IC here so this cell is self-contained
# (Uses noise_type, noise_amplitude, stripe_wavelength from experiment config)
clean_ic = base_ic_normalized.clone()

if noise_amplitude == 0:
    solver_ic_normalized = clean_ic
    print("Using clean IC (no noise)")
elif noise_type == "checkerboard":
    solver_ic_normalized = add_checkerboard_noise(clean_ic, amplitude=noise_amplitude)
elif noise_type == "diagonal_stripes":
    solver_ic_normalized = add_diagonal_stripes(clean_ic, amplitude=noise_amplitude, wavelength=stripe_wavelength)
elif noise_type == "high_freq_spectral":
    solver_ic_normalized = add_high_freq_spectral_noise(clean_ic, solver, amplitude=noise_amplitude, l_min=l_min_spectral)
elif noise_type == "gridscale_random":
    solver_ic_normalized = add_gridscale_random_noise(clean_ic, amplitude=noise_amplitude)
elif noise_type == "latitude_varying":
    solver_ic_normalized = add_latitude_varying_noise(clean_ic, amplitude=noise_amplitude, wavelength=stripe_wavelength)
else:
    solver_ic_normalized = clean_ic
    print(f"Unknown noise_type: {noise_type}, using clean IC")

# Convert to physical units and then to spectral
solver_ic_physical = solver_ic_normalized * torch.sqrt(inp_var) + inp_mean
solver_ic_spec = solver.grid2spec(solver_ic_physical)

# DIAGNOSTIC: Check how much noise survives the spectral roundtrip
solver_ic_roundtrip = solver.spec2grid(solver_ic_spec)
noise_before = (solver_ic_physical - (base_ic_grid)).std().item()
noise_after = (solver_ic_roundtrip - base_ic_grid).std().item()
print(f"Noise std BEFORE spectral conversion: {noise_before:.4f}")
print(f"Noise std AFTER spectral roundtrip:  {noise_after:.4f}")
if noise_before > 1e-9:
    print(f"Noise survival: {100*noise_after/noise_before:.1f}% (lmax={lmax} filters high-freq)")
else:
    print(f"Noise survival: N/A (noise approx zero)")
print(f"Using {noise_type} noise (amp={noise_amplitude})")

solver_steps_per_output = dt // dt_solver  # Should be 4 for dt=600

# Run solver and store PHYSICAL values (avoid normalization issues)
solver_results_physical = [solver.spec2grid(solver_ic_spec)]
current_spec = solver_ic_spec.clone()

print(f"Running solver for {n_solver_steps} steps (dt={dt}s, {solver_steps_per_output} sub-steps each)...")
print(f"Total simulation time: {n_solver_steps * dt / 3600:.1f} hours")

for step in range(n_solver_steps):
    current_spec = solver.timestep(current_spec, solver_steps_per_output)
    current_grid = solver.spec2grid(current_spec)
    solver_results_physical.append(current_grid)
    
    if not torch.isfinite(current_grid).all():
        print(f"  WARNING: Solver exploded at step {step+1}!")
        break

solver_results_physical = torch.stack(solver_results_physical, dim=0)

# Normalize for plotting (using consistent stats)
solver_results = (solver_results_physical - inp_mean) / torch.sqrt(inp_var)

print(f"Done. Results shape: {solver_results.shape}")
print(f"Physical - Initial: [{solver_results_physical[0].min():.1f}, {solver_results_physical[0].max():.1f}]")
print(f"Physical - Final:   [{solver_results_physical[-1].min():.1f}, {solver_results_physical[-1].max():.1f}]")
print(f"Normalized - Initial: [{solver_results[0].min():.2f}, {solver_results[0].max():.2f}]")
print(f"Normalized - Final:   [{solver_results[-1].min():.2f}, {solver_results[-1].max():.2f}]")

#%%
# ============ PLOT: SOLVER EVOLUTION ============
# Show IC, intermediate step, and final step
plot_time_idx = 2       # Which step to plot
channel_idx = 2

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

t0_data = solver_results[0, channel_idx].cpu().numpy()[lat_mask, :]
t_mid_data = solver_results[plot_time_idx, channel_idx].cpu().numpy()[lat_mask, :]
t_final_data = solver_results[-1, channel_idx].cpu().numpy()[lat_mask, :]

vmin = min(t0_data.min(), t_mid_data.min(), t_final_data.min())
vmax = max(t0_data.max(), t_mid_data.max(), t_final_data.max())

im0 = axes[0].pcolormesh(lon, lat[lat_mask], t0_data, cmap="viridis", vmin=vmin, vmax=vmax)
axes[0].set_title(f"t=0 (Noisy IC)")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].pcolormesh(lon, lat[lat_mask], t_mid_data, cmap="viridis", vmin=vmin, vmax=vmax)
axes[1].set_title(f"t={plot_time_idx}")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].pcolormesh(lon, lat[lat_mask], t_final_data, cmap="viridis", vmin=vmin, vmax=vmax)
axes[2].set_title(f"t={n_solver_steps} (Final)")
plt.colorbar(im2, ax=axes[2])

plt.suptitle(f"Solver Evolution: {channel_names[channel_idx]} with {noise_type} noise")
plt.tight_layout()
plt.show()

# Check if noise was damped
initial_noise_std = (solver_results[0] - clean).std().item()
final_noise_std = (solver_results[-1] - clean).std().item()  # Approximate - clean IC also evolved
print(f"Initial noise std: {initial_noise_std:.4f}")
print(f"Final deviation from original clean IC: {final_noise_std:.4f}")

#%%
# ============ COMPARE TO ML ARTIFACT (OPTIONAL) ============
# Load ML prediction to compare your synthetic noise to the real artifact
# Uncomment and run this cell after you find a noise pattern you like

# nc_path = "/glade/derecho/scratch/idavis/TH_SWE_output/disco_epd_W9_MB1_D66_L_morlet_validation_steps28.nc"
# ds = xr.open_dataset(nc_path)
# ml_pred = ds['prediction'][1, 24, channel_idx, :, :].values  # Sample 1, time 24
# ml_truth = ds['truth'][1, 24, channel_idx, :, :].values
# ml_artifact = ml_pred - ml_truth  # The "noise" the ML added

# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# artifact_max = np.abs(ml_artifact).max()
# axes[0].pcolormesh(lon, lat, ml_artifact, cmap="RdBu_r", vmin=-artifact_max, vmax=artifact_max)
# axes[0].set_title("ML Artifact (pred - truth)")
# axes[1].pcolormesh(lon, lat, noise_only[channel_idx].cpu().numpy(), cmap="RdBu_r", vmin=-artifact_max, vmax=artifact_max)
# axes[1].set_title(f"Synthetic: {noise_type}")
# plt.tight_layout()
# plt.show()

#%%

# %%
# ============ PLOT: SPHERICAL VIEW (CLEAN SOLVER) ============
if ccrs is not None:
    # Plot configuration
    plot_time_idx = -1       # Modify this to select timestep
    channel_idx = 1         # 0=h, 1=u, 2=v
    central_lat, central_lon = 45, 180
    
    # Get data from solver_results (which stores clean solver run)
    if plot_time_idx < len(solver_results_physical):
        data = solver_results_physical[plot_time_idx][channel_idx].cpu().numpy()
    else:
        print(f"Time index {plot_time_idx} out of range (max {len(solver_results_physical)-1})")
        data = solver_results_physical[-1][channel_idx].cpu().numpy()
    
    fig = plt.figure(figsize=(10, 8))
    proj = ccrs.Orthographic(central_latitude=central_lat, central_longitude=central_lon)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    
    lon = np.linspace(0, 360, nlon, endpoint=False)
    lat = np.linspace(90, -90, nlat)
    Lon, Lat = np.meshgrid(lon, lat)
    
    # Determine color limits
    vmin = data.min()
    vmax = data.max()
    abs_max = max(abs(vmin), abs(vmax))
    
    # Use divergent colormap for u/v, sequential for h
    cmap = "RdBu_r" if channel_idx > 0 else "viridis"
    if channel_idx > 0:
        vmin, vmax = -abs_max, abs_max
        
    im = ax.pcolormesh(Lon, Lat, data, cmap=cmap, transform=ccrs.PlateCarree(),
                       antialiased=True, vmin=vmin, vmax=vmax)
                       
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1,
                 color="gray", alpha=0.5, linestyle="--")
    
    ax.coastlines(color="black", alpha=0.3)
                 
    plt.colorbar(im, ax=ax, shrink=0.7, label=channel_names[channel_idx])
    plt.title(f"{channel_names[channel_idx]} at step {plot_time_idx} (dt={dt}s)\nSpherical View")
    plt.tight_layout()
    plt.show()
else:
    print("Cartopy not available, skipping spherical plot.")

# %%
