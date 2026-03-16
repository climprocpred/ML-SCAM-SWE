#!/usr/bin/env python3
"""
Generate a GIF animation from SWE NetCDF validation results.
Creates spherical plots for each timestep and combines them into an animated GIF.
"""
#%%
# ============ CONFIGURATION ============
# Edit these values to customize the animation

# Input NetCDF file path
input_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_filtered_diffusion_AR6_2_validation_steps104.nc"

# Output GIF path (None = auto-generate from input filename)
output_path = None

# Sample and channel to visualize
sample_idx = 1      # 0 = Galewsky IC, 1+ = random ICs
channel_idx = 1     # 0 = height, 1 = vorticity, 2 = divergence

# Animation settings
fps = 5             # Frames per second
start_time = 0      # Start timestep (0 = beginning)
end_time = None     # End timestep (None = all timesteps)
dpi = 150           # Resolution (higher = better quality, larger file)

# View settings
lat_cutoff = 90     # Latitude cutoff for display
central_lat = 30    # Central latitude for orthographic view
central_lon = 0     # Central longitude for orthographic view

# Color scaling
consistent_colorscale = True  # True = global limits, False = per-frame limits

# ============ END CONFIGURATION ============

import argparse
import os
import tempfile
import shutil
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import torch
from PIL import Image

# Cartopy for spherical projections
try:
    import cartopy.crs as ccrs
    import cartopy.feature
except ImportError:
    print("Error: cartopy is required. Install with: pip install cartopy")
    exit(1)


def get_projection(projection, central_latitude=0, central_longitude=0):
    """Get cartopy projection object."""
    if projection == "orthographic":
        proj = ccrs.Orthographic(central_latitude=central_latitude, central_longitude=central_longitude)
    elif projection == "robinson":
        proj = ccrs.Robinson(central_longitude=central_longitude)
    elif projection == "platecarree":
        proj = ccrs.PlateCarree(central_longitude=central_longitude)
    elif projection == "mollweide":
        proj = ccrs.Mollweide(central_longitude=central_longitude)
    else:
        raise ValueError(f"Unknown projection mode {projection}")
    return proj


def plot_sphere_custom(data, fig, ax, projection="robinson", cmap="RdBu", 
                       central_latitude=0, central_longitude=0, 
                       lon=None, lat=None, vmin=None, vmax=None, 
                       gridlines=False, **kwargs):
    """Plot data on a sphere with custom axes."""
    nlat = data.shape[-2]
    nlon = data.shape[-1]
    if lon is None:
        lon = np.linspace(0, 2 * np.pi, nlon + 1)[:-1]
    if lat is None:
        lat = np.linspace(np.pi / 2.0, -np.pi / 2.0, nlat)
    Lon, Lat = np.meshgrid(lon, lat)

    # Convert radians to degrees
    Lon = Lon * 180 / np.pi
    Lat = Lat * 180 / np.pi

    # Plot data
    im = ax.pcolormesh(Lon, Lat, data, cmap=cmap, transform=ccrs.PlateCarree(), 
                       antialiased=False, vmin=vmin, vmax=vmax, **kwargs)

    if gridlines:
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, 
                     linewidth=1, color="gray", alpha=0.6, linestyle="--")

    return im


def create_frame(ds, sample_idx, time_idx, channel_idx, lat_cutoff=90, 
                 central_lat=30, central_lon=0, global_vmin=None, global_vmax=None,
                 global_diff_max=None):
    """Create a single frame for the animation."""
    
    # Extract data
    pred = ds['prediction'][sample_idx, time_idx, channel_idx, :, :]
    truth = ds['truth'][sample_idx, time_idx, channel_idx, :, :]
    
    # Apply latitude cutoff
    pred = pred.where(abs(pred.lat) <= lat_cutoff)
    truth = truth.where(abs(truth.lat) <= lat_cutoff)
    diff = pred - truth
    
    # Convert to tensors
    pred_tensor = torch.tensor(pred.values)
    truth_tensor = torch.tensor(truth.values)
    diff_tensor = torch.tensor(diff.values)
    
    # Determine color limits
    if global_vmin is None:
        vmin = min(np.nanmin(pred.values), np.nanmin(truth.values))
    else:
        vmin = global_vmin
    if global_vmax is None:
        vmax = max(np.nanmax(pred.values), np.nanmax(truth.values))
    else:
        vmax = global_vmax
    if global_diff_max is None:
        diff_max = np.nanmax(np.abs(diff.values))
    else:
        diff_max = global_diff_max
    
    # Create figure
    fig = plt.figure(figsize=(18, 6))
    
    # Plot prediction
    proj = get_projection("orthographic", central_latitude=central_lat, central_longitude=central_lon)
    ax1 = fig.add_subplot(1, 3, 1, projection=proj)
    im_pred = plot_sphere_custom(pred_tensor, fig, ax1, vmin=vmin, vmax=vmax, 
                                  central_latitude=central_lat, gridlines=True, 
                                  projection="orthographic", cmap="viridis")
    ax1.set_title(f"Prediction (t={time_idx})")
    
    # Plot truth
    ax2 = fig.add_subplot(1, 3, 2, projection=proj)
    im_truth = plot_sphere_custom(truth_tensor, fig, ax2, vmin=vmin, vmax=vmax,
                                   central_latitude=central_lat, gridlines=True,
                                   projection="orthographic", cmap="viridis")
    ax2.set_title(f"Truth (t={time_idx})")
    
    # Plot difference
    ax3 = fig.add_subplot(1, 3, 3, projection=proj)
    im_diff = plot_sphere_custom(diff_tensor, fig, ax3, vmin=-diff_max, vmax=diff_max,
                                  central_latitude=central_lat, gridlines=True,
                                  projection="orthographic", cmap="RdBu_r")
    ax3.set_title("Difference")
    
    # Channel names
    channel_names = ['h (height)', 'vorticity', 'divergence']
    var_name = channel_names[channel_idx] if channel_idx < len(channel_names) else f"Channel {channel_idx}"
    
    plt.suptitle(f"Sample {sample_idx}, Time {time_idx}, {var_name}")
    
    # Add colorbars
    cbar_ax_main = fig.add_axes([0.08, 0.12, 0.55, 0.04])
    cbar_main = fig.colorbar(im_pred, cax=cbar_ax_main, orientation='horizontal')
    cbar_main.set_label('Prediction / Truth', fontsize=12)
    cbar_main.ax.tick_params(labelsize=10)
    
    cbar_ax_diff = fig.add_axes([0.66, 0.12, 0.25, 0.04])
    cbar_diff = fig.colorbar(im_diff, cax=cbar_ax_diff, orientation='horizontal')
    cbar_diff.set_label('Difference', fontsize=12)
    cbar_diff.ax.tick_params(labelsize=10)
    
    plt.subplots_adjust(bottom=0.18)
    
    return fig


def compute_global_limits(ds, sample_idx, channel_idx, lat_cutoff=90):
    """Compute global min/max across all timesteps for consistent color scaling."""
    n_times = ds.dims['time']
    
    all_vmin = []
    all_vmax = []
    all_diff_max = []
    
    for t in range(n_times):
        pred = ds['prediction'][sample_idx, t, channel_idx, :, :]
        truth = ds['truth'][sample_idx, t, channel_idx, :, :]
        
        pred = pred.where(abs(pred.lat) <= lat_cutoff)
        truth = truth.where(abs(truth.lat) <= lat_cutoff)
        diff = pred - truth
        
        all_vmin.append(min(np.nanmin(pred.values), np.nanmin(truth.values)))
        all_vmax.append(max(np.nanmax(pred.values), np.nanmax(truth.values)))
        all_diff_max.append(np.nanmax(np.abs(diff.values)))
    
    return min(all_vmin), max(all_vmax), max(all_diff_max)


def create_gif(input_path, output_path, sample_idx=0, channel_idx=0, 
               fps=5, lat_cutoff=90, central_lat=30, central_lon=0,
               start_time=0, end_time=None, consistent_colorscale=True, dpi=150):
    """Create animated GIF from NetCDF file."""
    
    print(f"Loading {input_path}...")
    ds = xr.open_dataset(input_path)
    
    n_times = ds.dims['time']
    if end_time is None:
        end_time = n_times
    end_time = min(end_time, n_times)
    
    print(f"Creating animation for sample {sample_idx}, channel {channel_idx}")
    print(f"Time range: {start_time} to {end_time} ({end_time - start_time} frames)")
    
    # Compute global limits for consistent color scaling
    if consistent_colorscale:
        print("Computing global color limits...")
        global_vmin, global_vmax, global_diff_max = compute_global_limits(
            ds, sample_idx, channel_idx, lat_cutoff
        )
        print(f"  Value range: [{global_vmin:.4f}, {global_vmax:.4f}]")
        print(f"  Max difference: {global_diff_max:.4f}")
    else:
        global_vmin, global_vmax, global_diff_max = None, None, None
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    frame_paths = []
    
    try:
        # Generate frames
        for t in range(start_time, end_time):
            print(f"\rGenerating frame {t+1}/{end_time}...", end="", flush=True)
            
            fig = create_frame(
                ds, sample_idx, t, channel_idx, 
                lat_cutoff=lat_cutoff,
                central_lat=central_lat, 
                central_lon=central_lon,
                global_vmin=global_vmin,
                global_vmax=global_vmax,
                global_diff_max=global_diff_max
            )
            
            frame_path = os.path.join(temp_dir, f"frame_{t:04d}.png")
            fig.savefig(frame_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            frame_paths.append(frame_path)
        
        print("\nAssembling GIF...")
        
        # Load frames and create GIF
        frames = [Image.open(fp) for fp in frame_paths]
        
        # Calculate duration in milliseconds from fps
        duration = int(1000 / fps)
        
        # Save as GIF with high quality settings
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,  # 0 means infinite loop
            optimize=False  # Disable optimization for better color quality
        )
        
        print(f"GIF saved to: {output_path}")
        print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        print(f"  Frames: {len(frames)}")
        print(f"  FPS: {fps}")
        
    finally:
        # Cleanup temporary directory
        shutil.rmtree(temp_dir)
    
    ds.close()


# ============ RUN GIF CREATION ============
# Generate default output path if not specified
if output_path is None:
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = f"{base_name}_sample{sample_idx}_ch{channel_idx}.gif"

# Create the GIF
create_gif(
    input_path=input_path,
    output_path=output_path,
    sample_idx=sample_idx,
    channel_idx=channel_idx,
    fps=fps,
    lat_cutoff=lat_cutoff,
    central_lat=central_lat,
    dpi=dpi,
    central_lon=central_lon,
    start_time=start_time,
    end_time=end_time,
    consistent_colorscale=consistent_colorscale
)

# %%
