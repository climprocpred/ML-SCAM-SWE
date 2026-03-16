#%%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch_harmonics.plotting import plot_sphere

# %%
# Load the dataset
# Update this path to your specific output file
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/EPD_W9_MB1_D66_L_validation_steps24.nc"
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/disco_epd_W9_MB1_D66_L_morlet_validation_steps28.nc"
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/disco_epd_W9_MB1_D66_L_morlet_filtered_rollout_steps192.nc"
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/disco_epd_W9_MB1_D66_L_morlet_filtered_rollout_steps240.nc" # AR12
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/disco_epd_W9_MB1_D66_L_morlet_filtered_rollout_steps241.nc" @# R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/disco_epd_W9_MB1_D66_L_morlet_filtered_rollout_steps500.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/control_validation_steps92.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_validation_steps92.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/filtered_validation_steps92.nc" # R 6

# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_filtered_validation_steps92.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/zernike_validation_steps92.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_filtered_zernike_validation_steps92.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/disco_mlp_transformer_validation_steps92.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/disco_mlp_transformer_e128_validation_steps92.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_filtered_norm2_validation_steps92.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_filtered_2L_e126_validation_steps92.nc" # R 6
# # ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_filtered_e34_validation_steps92.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_filtered_e34_validation_steps92.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_filtered_e34_4L_validation_steps92.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_filtered_W7_K7_validation_steps92.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/disco_epd_W9_MB1_D66_L_morlet_filtered_rollout_steps500.nc" # R 6
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_filtered_BHMLP_validation_steps92.nc" # R 6
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/sfno_sc2_layers4_e32_validation_steps92.nc" # R 6
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/hierarchical_disco_epd_ctrl_validation_steps92.nc" # R 6
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/sfno_sc2_layers4_e32_100_validation_steps92.nc" # R 6
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/lsno_sc2_layers4_e32_validation_steps92.nc" # R 6
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/control_validation_steps92.nc" # R 6
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_filtered_validation_steps92.nc" # R 6
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_filtered_validation_steps121.nc" # R 6
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_filtered_validation_steps122.nc" # R 6
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_filtered_diffusion_AR6_2_validation_steps93.nc" # R 6
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_filtered_diffusion_AR6_2_validation_steps94.nc" # R 6
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_filtered_diffusion_AR6_2_validation_steps95.nc" # R 6
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_filtered_diffusion_AR6_2_validation_steps104.nc" # R 6
# # ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_filtered_diffusion_curriculum_rollout_T1_validation_steps101.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_filtered_diffusion_curriculum_rollout_T2_validation_steps103.nc" # R 6
# # ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_curriculum_rollout_T2_validation_steps103.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_filtered_diffusion_curriculum_reg_sched_validation_steps91.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_laplacian_validation_steps91.nc" # R 6
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_laplacian_8_validation_steps87.nc" # R 6
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_laplacian_8_post_AR24_validation_steps87.nc" # R 6
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_laplacian_8_post_AR24_mid_cur_validation_steps87.nc" # R 6
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_laplacian_8_cons_T1_validation_steps87.nc" # R 6
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_laplacian_8_validation_steps93.nc" # R 6
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_laplacian_8_mid_AR36_validation_steps87.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_laplacian_8_CCC_tuned_hard_cons_L7_epoch1_validation_steps92.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_laplacian_8_CCC_tuned_hard_cons_L7_epoch2_validation_steps92.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_laplacian_8_CCC_tuned_hard_cons_L7_epoch4_validation_steps92.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_laplacian_8_enforce_conservation_no_train_validation_steps92.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_L7_MC5_H3_AR36_validation_steps92.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_history3_conservation_CCC7_l23_AR12_validation_steps92.nc" # R 6

# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_conservation_CCC0_l21_AR24_validation_steps92.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_conservation_CCC7_l23_AR36_validation_steps92.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_history3_conservation_CCC7_l23_AR36_validation_steps92.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_history3_conservation_CCC7_l23_AR36_recent_validation_steps92.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_history3_conservation_CCC7_l23_AR36_L7_epoch1_validation_steps92.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_history3_conservation_CCC7_l23_AR36_L7_epoch3_validation_steps92.nc" # R 6

# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output//film_history3_conservation_CCC7_l23_AR36_L7_epoch3_UV3_validation_steps92.nc" # R 6

# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_laplacian_8_CCC_tuned_hard_cons_L7-UV_L0.4_epoch2_validation_steps92.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_laplacian_8_CCC_tuned_hard_cons_L7-UV_L0.4_epoch3_validation_steps92.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_laplacian_8_CCC_tuned_hard_cons_L7-UV_L0.4_base_validation_steps92.nc" # R 6
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_laplacian_8_validation_steps92.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_laplacian_8_CCC_tuned_hard_cons_L7-UV_L0.4_epoch4_filtered_validation_steps92.nc" # R 6
# ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_laplacian_8_CCC_tuned_hard_cons_L7-UV_L0.4_base_filtered_validation_steps92.nc" # R 6

ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_laplacian_8_AR2_validation_steps92.nc"
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_laplacian_8_delete_me_validation_steps92.nc"
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_laplacian_8_AR4_validation_steps92.nc"
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_laplacian_8_AR6_validation_steps92.nc"
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_laplacian_8_AR12_validation_steps92.nc"
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_laplacian_8_AR24_validation_steps92.nc"
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_laplacian_8_AR36_validation_steps92.nc"
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_laplacian_8_AR48_validation_steps92.nc"
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/film_laplacian_8_AR96_validation_steps92.nc"

ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/all_tricks_2_AR24_validation_steps92.nc"
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/s2ntransformer_sc2_layers4_e128_validation_steps92.nc"
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/sfno_sc2_layers4_e32_validation_steps92.nc"
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/s2ntransformer_sc2_layers4_e128_control2_validation_steps183.nc"
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/s2ntransformer_stochastic_ic_mach_validation_steps92.nc"
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/s2ntransformer_stochastic_ic_mach_llimit_AR2_validation_steps92.nc"
ds_path = "/glade/derecho/scratch/idavis/TH_SWE_output/s2ntransformer_stochastic_all_tricks2_AR4_validation_steps92.nc"
ds = xr.open_dataset(ds_path)
    # print(ds)

# Configuration
#%%
sample_idx = 1
time_idx = 8
lat_cutoff = 90
channel_idx = 2 # 0: h (height), 1: vorticity, 2: divergence
var_name = f"Channel {channel_idx}"

# Check bounds
if time_idx >= ds.dims['time']:
    print(f"Time index {time_idx} out of bounds, setting to last step.")
    time_idx = ds.dims['time'] - 1

# Extract data
# shape: (sample, time, channel, lat, lon)
pred = ds['prediction'][sample_idx, time_idx, channel_idx, :, :]
truth = ds['truth'][sample_idx, time_idx, channel_idx, :, :]

# truth = truth.where(np.abs(truth) < 0.005, 0.0) # Threshold lowered slightly to catch tails
# pred = pred.where(np.abs(pred) < 0.01, 0.0)
pred = pred.sel(lat=slice(-lat_cutoff, lat_cutoff))
truth = truth.sel(lat=slice(-lat_cutoff, lat_cutoff))
diff = pred - truth

# Plotting Rectilinear
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Prediction
pred.plot(ax=axes[0])
axes[0].set_title(f"Prediction (t={time_idx})")

# Truth
truth.plot(ax=axes[1])
axes[1].set_title(f"Ground Truth (t={time_idx})")

# Difference
diff.plot(ax=axes[2], cmap="RdBu_r")
axes[2].set_title(f"Difference (Pred - Truth)")

plt.suptitle(f"Sample {sample_idx}, Time {time_idx}, {var_name}")
plt.tight_layout()

# Save figure
save_path = "swe_plot_output.png"
plt.savefig(save_path)
print(f"Rectilinear plot saved to {save_path}")

plt.show() 

# %%
# Spherical Plotting
# Configuration
sample_idx = 3
time_idx =  -1
lat_cutoff = 90
channel_idx = 1 #0: h (height), 1: vorticity, 2: divergence 3: u, 4: v
var_name = f"Channel {channel_idx}"
central_lat = 45 # Latitude of the view center
central_lon = 180 # Longitude of the view center


# Extract data
# shape: (sample, time, channel, lat, lon)
pred = ds['prediction'][sample_idx, time_idx, channel_idx, :, :]
truth = ds['truth'][sample_idx, time_idx, channel_idx, :, :]

# Apply latitude cutoff for spherical plotting (fill with NaN)
pred = pred.where(abs(pred.lat) <= lat_cutoff)
truth = truth.where(abs(truth.lat) <= lat_cutoff)

diff = pred - truth
print("Generating spherical plots...")

# Convert xarray to tensor for plot_sphere
# plot_sphere expects [lat, lon]
pred_tensor = torch.tensor(pred.values)
truth_tensor = torch.tensor(truth.values)
diff_tensor = torch.tensor(diff.values)

fig = plt.figure(figsize=(18, 6))

# Helper functions adapted from torch_harmonics.plotting to allow passing 'ax'
try:
    import cartopy.crs as ccrs
    import cartopy.feature
except ImportError:
    print("Warning: cartopy not installed, spherical plots might fail.")
    ccrs = None

def get_projection(projection, central_latitude=0, central_longitude=0):
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

def plot_sphere_custom(data, fig, ax, projection="robinson", cmap="RdBu", title=None, colorbar=False, coastlines=False, gridlines=False, central_latitude=0, central_longitude=0, lon=None, lat=None, vmin=None, vmax=None, **kwargs):
    if ccrs is None:
        return
        
    nlat = data.shape[-2]
    nlon = data.shape[-1]
    if lon is None:
        lon = np.linspace(0, 2 * np.pi, nlon + 1)[:-1]
    if lat is None:
        lat = np.linspace(np.pi / 2.0, -np.pi / 2.0, nlat)
    Lon, Lat = np.meshgrid(lon, lat)

    # convert radians to degrees
    Lon = Lon * 180 / np.pi
    Lat = Lat * 180 / np.pi

    # contour data over the map.
    im = ax.pcolormesh(Lon, Lat, data, cmap=cmap, transform=ccrs.PlateCarree(), antialiased=False, vmin=vmin, vmax=vmax, **kwargs)

    # add features if requested
    if coastlines:
        ax.add_feature(cartopy.feature.COASTLINE, edgecolor="white", facecolor="none", linewidth=1.5)

    # add colorbar if requested
    if colorbar:
        plt.colorbar(im, ax=ax)

    # add gridlines
    if gridlines:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color="gray", alpha=0.6, linestyle="--")

    return im

# Helper to plot subplot
def plot_sub(tensor, index, title, vmin=None, vmax=None, cmap="viridis"):
    # Create subplot with appropriate projection
    # NOTE: We must create the axes with projection HERE, passing it to add_subplot
    proj = get_projection("orthographic", central_latitude=central_lat, central_longitude=central_lon)
    ax = fig.add_subplot(1, 3, index, projection=proj)
    
    im = plot_sphere_custom(
        tensor, 
        fig, 
        ax=ax,
        vmax=vmax, 
        vmin=vmin, 
        central_latitude=central_lat, 
        central_longitude=central_lon,
        gridlines=True, 
        projection="orthographic",
        cmap=cmap,
        title=title
    )
    ax.set_title(title)
    return im, ax

# Determine limits from data for consistent comparison
# Use numpy nanmin/nanmax to ignore NaNs from the latitude cutoff
vmin = min(np.nanmin(pred.values), np.nanmin(truth.values))
vmax = max(np.nanmax(pred.values), np.nanmax(truth.values))

im_pred, ax_pred = plot_sub(pred_tensor, 1, f"Prediction (t={time_idx})", vmin=vmin, vmax=vmax)
im_truth, ax_truth = plot_sub(truth_tensor, 2, f"Truth (t={time_idx})", vmin=vmin, vmax=vmax)

# Difference often needs its own centered scale
diff_max = np.nanmax(np.abs(diff.values))
im_diff, ax_diff = plot_sub(diff_tensor, 3, f"Difference", vmin=-diff_max, vmax=diff_max, cmap="RdBu_r")

plt.suptitle(f"Spherical View: Sample {sample_idx}, Time {time_idx}, {var_name}")

# Add shared colorbar for prediction and truth (spans first two subplots)
cbar_ax_main = fig.add_axes([0.08, 0.12, 0.55, 0.04])  # [left, bottom, width, height]
cbar_main = fig.colorbar(im_pred, cax=cbar_ax_main, orientation='horizontal')
cbar_main.set_label('Prediction / Truth', fontsize=12)
cbar_main.ax.tick_params(labelsize=10)

# Add colorbar for difference plot
cbar_ax_diff = fig.add_axes([0.66, 0.12, 0.25, 0.04])
cbar_diff = fig.colorbar(im_diff, cax=cbar_ax_diff, orientation='horizontal')
cbar_diff.set_label('Difference', fontsize=12)
cbar_diff.ax.tick_params(labelsize=10)

plt.subplots_adjust(bottom=0.18)  # Make room for colorbars

save_path_sphere = "swe_plot_sphere.png"
plt.savefig(save_path_sphere)
print(f"Spherical plot saved to {save_path_sphere}")

plt.show()

# %%
# Area-weighted RMSE vs Lead Time for each variable
# Compute cosine(latitude) weights for proper spherical area weighting

# Extract model name from ds_path
import re
model_name = os.path.basename(ds_path)
model_name = re.sub(r'_(validation|rollout)_steps\d+\.nc$', '', model_name)

ds = ds.isel(time=slice(0,93))

# w2 = np.arange(0,193)
# w2 = 1.003**w2
# ds["prediction"][:,:,2] /= w2[None,:, None,None]

# Get lat values in radians
lat_rad = np.deg2rad(ds['lat'].values)
cos_weights = np.cos(lat_rad)

# Normalize weights so they sum to 1
cos_weights = cos_weights / cos_weights.sum()

# Reshape weights for broadcasting: (lat,) -> (1, 1, 1, lat, 1)
weights = cos_weights[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]

# Get prediction and truth: (sample, time, channel, lat, lon)
pred_all = ds['prediction'].values
truth_all = ds['truth'].values

# Compute squared error
squared_error = (pred_all - truth_all) ** 2

# Area-weighted mean over lat and lon, then mean over samples
# weighted_mse shape: (sample, time, channel)
weighted_mse = (squared_error * weights).sum(axis=-2).mean(axis=-1)  # Sum over lat, mean over lon
weighted_mse = weighted_mse.mean(axis=0)  # Mean over samples -> (time, channel)

# Calculate truth temporal variance and standard deviation for normalization
# truth_all shape: (sample, time, channel, lat, lon)
# weights shape: (1, 1, 1, lat, 1)

# 1. Compute temporal mean per location (averaged over samples)
# Result shape: (channel, lat, lon)
truth_time_mean = truth_all.mean(axis=(0, 1))

# Broadcast for subtraction: (1, 1, channel, lat, lon)
truth_time_mean_expanded = truth_time_mean[np.newaxis, np.newaxis, ...]

# 2. Compute temporal variance at each location
# (truth - temporal_mean)^2
# Averaged over sample and time dims
# Result shape: (channel, lat, lon)
truth_temp_var_field = ((truth_all - truth_time_mean_expanded) ** 2).mean(axis=(0, 1))

# 3. Aggregate spatially (area-weighted average) to get single scalar variance per channel
# weights shape needs to be (channel, lat, 1) or broadcastable
# weights is (1, 1, 1, lat, 1) -> squeeze to (lat, 1)
weights_2d = weights.squeeze() # (lat,)
weights_broad = weights_2d[np.newaxis, :, np.newaxis] # (1, lat, 1)

# Result shape: (channel,)
truth_temp_avg_var = (truth_temp_var_field * weights_broad).sum(axis=(-2, -1)) / weights_broad.sum()
truth_std = np.sqrt(truth_temp_avg_var)

print(f"Truth Temporal Standard Deviations shape: {truth_std.shape}")
print(f"Truth Temporal Standard Deviations: {truth_std}")

# RMSE
rmse = np.sqrt(weighted_mse)
# Normalize RMSE by temporal std
rmse_normalized = rmse / truth_std[None, :]

# Get time/channel info
n_times = rmse.shape[0]
n_channels = rmse.shape[1]
channel_names = ['h (height)', 'vorticity', 'divergence', 'u', 'v'][:n_channels]

# Plot RMSE vs lead time
fig, ax = plt.subplots(figsize=(10, 6))

for ch_idx in range(n_channels):
    # ax.plot(range(n_times), rmse[:, ch_idx], label=channel_names[ch_idx], linewidth=2)
    ax.plot(range(n_times), rmse_normalized[:, ch_idx], label=channel_names[ch_idx], linewidth=2)

ax.set_xlabel('Lead Time (steps)', fontsize=12)
ax.set_ylabel('Normalized RMSE (RMSE / Temporal Std Dev)', fontsize=12)
ax.set_title(f'Normalized RMSE vs Lead Time: {model_name}', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.ylim(0, 0.2) # Adjusted limit for normalized values
plt.tight_layout()
plt.savefig('rmse_vs_leadtime.png', dpi=150)
print("RMSE plot saved to rmse_vs_leadtime.png")
plt.show()

# Print summary table
print("\nNormalized RMSE Summary:")
print("-" * 50)
print(f"{'Lead Time':<12} " + " ".join([f"{name:<15}" for name in channel_names]))
print("-" * 50)
for t in range(0, n_times, max(1, n_times // 10)):  # Print ~10 evenly spaced rows
    # row = f"{t:<12} " + " ".join([f"{rmse[t, ch]:<15.6f}" for ch in range(n_channels)])
    row = f"{t:<12} " + " ".join([f"{rmse_normalized[t, ch]:<15.6f}" for ch in range(n_channels)])
    print(row)
print("-" * 50)
print(f"{'Final':<12} " + " ".join([f"{rmse_normalized[-1, ch]:<15.6f}" for ch in range(n_channels)]))

# %%
# Variance/Mean vs Lead Time for each variable
# Shows variance growth/decay or mean drift over autoregressive rollout

# Toggle: "variance" or "mean"
plot_mode = "variance"  # Change to "mean" to plot global mean instead

# Reload ds without the time slice applied above
ds_var = xr.open_dataset(ds_path)
ds_var = ds_var.isel(time=slice(0,30))

import re
model_name = os.path.basename(ds_path)
model_name = re.sub(r'_(validation|rollout)_steps\d+\.nc$', '', model_name)

# Get lat values in radians for area weighting
lat_rad = np.deg2rad(ds_var['lat'].values)
cos_weights = np.cos(lat_rad)
cos_weights = cos_weights / cos_weights.sum()

# Reshape weights for broadcasting: (lat,) -> (1, 1, 1, lat, 1)
weights = cos_weights[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]

# Get prediction and truth: (sample, time, channel, lat, lon)
pred_all = ds_var['prediction'].values
truth_all = ds_var['truth'].values

# Compute area-weighted mean for each sample/time/channel
# Shape: (sample, time, channel)
pred_mean = (pred_all * weights).sum(axis=(-2, -1)) / pred_all.shape[-1]  # Sum over lat (weighted), mean over lon
truth_mean = (truth_all * weights).sum(axis=(-2, -1)) / truth_all.shape[-1]

# Compute area-weighted variance: E[(x - mean)^2]
pred_var = ((pred_all - pred_mean[..., np.newaxis, np.newaxis])**2 * weights).sum(axis=-2).mean(axis=-1)
truth_var = ((truth_all - truth_mean[..., np.newaxis, np.newaxis])**2 * weights).sum(axis=-2).mean(axis=-1)

# Average over samples -> (time, channel)
pred_var_avg = pred_var.mean(axis=0)
truth_var_avg = truth_var.mean(axis=0)
pred_mean_avg = pred_mean.mean(axis=0)
truth_mean_avg = truth_mean.mean(axis=0)

# Select data based on mode
if plot_mode == "variance":
    # Normalize variance by time-averaged truth variance
    norm_factor = truth_var_avg.mean(axis=0) # (channel,)
    pred_data = pred_var_avg / norm_factor[None, :]
    truth_data = truth_var_avg / norm_factor[None, :]
    ylabel = 'Normalized Variance (Var / Mean Truth Var)'
    title_suffix = 'Normalized Variance'
    save_name = 'variance_vs_leadtime.png'
else:  # mean
    pred_data = pred_mean_avg
    truth_data = truth_mean_avg
    ylabel = 'Area-Weighted Mean'
    title_suffix = 'Mean'
    save_name = 'mean_vs_leadtime.png'

# Get time/channel info
n_times = pred_data.shape[0]
n_channels = pred_data.shape[1]
channel_names = ['h (height)', 'vorticity', 'divergence', 'u', 'v'][:n_channels]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

for ch_idx in range(n_channels):
    ax.plot(range(n_times), pred_data[:, ch_idx], 
            label=f'{channel_names[ch_idx]} (forecast)', 
            linewidth=2, color=colors[ch_idx], linestyle='-', alpha=0.7)
    ax.plot(range(n_times), truth_data[:, ch_idx], 
            label=f'{channel_names[ch_idx]} (truth)', 
            linewidth=2, color=colors[ch_idx], linestyle='--', alpha=0.7)

ax.set_xlabel('Lead Time (steps)', fontsize=12)
ax.set_ylabel(ylabel, fontsize=12)
ax.set_title(f'{title_suffix} vs Lead Time: {model_name}', fontsize=14)
ax.legend(fontsize=10, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(save_name, dpi=150)
print(f"{title_suffix} plot saved to {save_name}")
plt.show()

# Print ratio at final step
print(f"\n{title_suffix} Ratio (Forecast/Truth) at Final Step:")
print("-" * 50)
for ch_idx in range(n_channels):
    if plot_mode == "variance":
        ratio = pred_data[-1, ch_idx] / truth_data[-1, ch_idx]
        print(f"{channel_names[ch_idx]}: {ratio:.4f}")
    else:
        diff = pred_data[-1, ch_idx] - truth_data[-1, ch_idx]
        print(f"{channel_names[ch_idx]}: pred={pred_data[-1, ch_idx]:.4f}, truth={truth_data[-1, ch_idx]:.4f}, diff={diff:.4f}")

# %%
