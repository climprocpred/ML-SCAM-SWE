"""
Plotting utilities for SWE model evaluation.

All functions accept raw numpy arrays (or torch tensors), return matplotlib Figure
objects, and have NO wandb dependency. The caller can:
  - wandb.log({"name": wandb.Image(fig)})
  - fig.savefig("name.png")
  - plt.show()

Expected data shapes (matching autoregressive_inference output):
  preds, truth: (n_samples, n_times, n_channels, nlat, nlon)
  - n_channels typically 3 (height, vorticity, divergence) or 5 (+u, +v)
  - Data is assumed to be normalized (zero-mean, unit-variance per channel)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Defaults ────────────────────────────────────────────────────────────────────

CHANNEL_NAMES = ['h (height)', 'vorticity', 'divergence', 'u', 'v']
CHANNEL_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']


def _to_numpy(x):
    """Convert torch tensor to numpy if needed."""
    if hasattr(x, 'numpy'):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _cos_weights(nlat):
    """Cosine latitude weights for area-weighted statistics, normalized to sum to 1."""
    lat_rad = np.linspace(np.pi / 2, -np.pi / 2, nlat)
    w = np.cos(lat_rad)
    return w / w.sum()


# ── 1. Rollout Snapshots ────────────────────────────────────────────────────────

def plot_rollout_snapshots(
    preds,
    truth,
    channel=0,
    sample=1,
    time_indices=None,
    n_snapshots=6,
    title=None,
    lat_range=None,
    figsize=None,
    cmap_field='viridis',
    cmap_diff='RdBu_r',
):
    """
    Rectilinear multi-timestep pred / truth / diff panels.

    Parameters
    ----------
    preds, truth : array-like, shape (n_samples, n_times, n_channels, nlat, nlon)
    channel : int
        Which channel to plot (0=height, 1=vorticity, 2=divergence, ...).
    sample : int
        Which sample (IC) to plot.
    time_indices : list of int, optional
        Specific timestep indices to show. If None, evenly spaced.
    n_snapshots : int
        Number of snapshots if time_indices is None.
    title : str, optional
        Suptitle. Auto-generated if None.
    lat_range : tuple (lat_min, lat_max), optional
        Latitude slice in degrees (e.g. (-48, 48)). None = full globe.
    figsize : tuple, optional
    cmap_field : str
        Colormap for prediction/truth panels.
    cmap_diff : str
        Colormap for difference panels.

    Returns
    -------
    matplotlib.Figure
    """
    preds = _to_numpy(preds)
    truth = _to_numpy(truth)

    n_times = preds.shape[1]
    nlat = preds.shape[3]
    nlon = preds.shape[4]

    # Pick timesteps
    if time_indices is None:
        time_indices = np.linspace(0, n_times - 1, n_snapshots, dtype=int)
    n_snap = len(time_indices)

    # Coordinate arrays
    lats = np.linspace(90, -90, nlat)
    lons = np.linspace(0, 360, nlon, endpoint=False)

    # Latitude slice
    if lat_range is not None:
        lat_mask = (lats >= lat_range[0]) & (lats <= lat_range[1])
    else:
        lat_mask = np.ones(nlat, dtype=bool)

    lats_sub = lats[lat_mask]

    # Extract data for this sample/channel
    p = preds[sample, :, channel, :, :][:, lat_mask, :]  # (times, lat_sub, lon)
    t = truth[sample, :, channel, :, :][:, lat_mask, :]

    # Global colorbar limits (across all snapshots, pred and truth)
    vmin = min(p[time_indices].min(), t[time_indices].min())
    vmax = max(p[time_indices].max(), t[time_indices].max())

    # Global difference limit (so all difference plots share same scale)
    # We want consistent scaling across time to show error growth
    diff_all = p[time_indices] - t[time_indices]
    diff_max_global = max(abs(diff_all.min()), abs(diff_all.max()), 1e-12)

    # Layout: rows = timesteps, cols = [pred, truth, diff]
    if figsize is None:
        figsize = (14, 2.5 * n_snap + 0.8)

    fig, axes = plt.subplots(n_snap, 3, figsize=figsize,
                             gridspec_kw={'wspace': 0.08, 'hspace': 0.35})
    if n_snap == 1:
        axes = axes[np.newaxis, :]  # ensure 2D

    ch_name = CHANNEL_NAMES[channel] if channel < len(CHANNEL_NAMES) else f'ch{channel}'

    for row, ti in enumerate(time_indices):
        pred_field = p[ti]
        truth_field = t[ti]
        diff_field = pred_field - truth_field
        # diff_max = max(abs(diff_field.min()), abs(diff_field.max()), 1e-12)

        # Prediction
        im_p = axes[row, 0].pcolormesh(lons, lats_sub, pred_field,
                                        cmap=cmap_field, vmin=vmin, vmax=vmax,
                                        shading='auto')
        axes[row, 0].set_ylabel(f't={ti}', fontsize=10, fontweight='bold')

        # Truth
        im_t = axes[row, 1].pcolormesh(lons, lats_sub, truth_field,
                                        cmap=cmap_field, vmin=vmin, vmax=vmax,
                                        shading='auto')

        # Difference
        im_d = axes[row, 2].pcolormesh(lons, lats_sub, diff_field,
                                        cmap=cmap_diff, vmin=-diff_max_global, vmax=diff_max_global,
                                        shading='auto')

        # Tick cleanup
        for col in range(3):
            axes[row, col].set_xticks([])
            if col > 0:
                axes[row, col].set_yticks([])

    # Column headers
    axes[0, 0].set_title('Prediction', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Truth', fontsize=11, fontweight='bold')
    axes[0, 2].set_title('Difference', fontsize=11, fontweight='bold')

    # Colorbars
    fig.colorbar(im_p, ax=axes[:, :2].ravel().tolist(), shrink=0.6, pad=0.02,
                 label=ch_name)
    fig.colorbar(im_d, ax=axes[:, 2].ravel().tolist(), shrink=0.6, pad=0.02,
                 label='Pred − Truth')

    if title is None:
        title = f'Rollout Snapshots — {ch_name} (sample {sample})'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.95)

    return fig


# ── 2. Normalized RMSE vs Lead Time ─────────────────────────────────────────────

def plot_rmse_vs_leadtime(
    preds,
    truth,
    channels=None,
    dt_minutes=10.0,
    ylim=None,
    title=None,
    figsize=(10, 6),
):
    """
    Area-weighted, temporally-normalized RMSE vs lead time for each variable.

    RMSE is normalized by the time-averaged, area-weighted standard deviation
    of the truth field for each channel (so all variables are comparable).

    Parameters
    ----------
    preds, truth : array-like, shape (n_samples, n_times, n_channels, nlat, nlon)
    channels : list of int, optional
        Which channels to plot. None = all.
    dt_minutes : float
        Time step in minutes between successive lead times (for x-axis label).
    ylim : tuple, optional
        Y-axis limits. None = auto.
    title : str, optional
    figsize : tuple

    Returns
    -------
    matplotlib.Figure
    """
    preds = _to_numpy(preds)
    truth = _to_numpy(truth)

    n_samples, n_times, n_channels, nlat, nlon = preds.shape

    if channels is None:
        channels = list(range(n_channels))

    # Area weights: (1, 1, 1, lat, 1)
    cos_w = _cos_weights(nlat)
    weights = cos_w[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]

    # Squared error → area-weighted mean over lat/lon → mean over samples
    sq_err = (preds - truth) ** 2
    # Sum over lat (weighted), mean over lon → (sample, time, channel)
    weighted_mse = (sq_err * weights).sum(axis=-2).mean(axis=-1)
    weighted_mse = weighted_mse.mean(axis=0)  # (time, channel)

    rmse = np.sqrt(weighted_mse)

    # Normalization: temporal std dev of truth (area-weighted)
    # 1. Temporal mean per location: (channel, lat, lon)
    truth_time_mean = truth.mean(axis=(0, 1))
    # 2. Temporal variance field: (channel, lat, lon)
    truth_temp_var = ((truth - truth_time_mean[np.newaxis, np.newaxis]) ** 2).mean(axis=(0, 1))
    # 3. Area-weighted average → (channel,)
    w_2d = cos_w[np.newaxis, :, np.newaxis]  # (1, lat, 1)
    truth_avg_var = (truth_temp_var * w_2d).sum(axis=(-2, -1)) / w_2d.sum()
    truth_std = np.sqrt(truth_avg_var)

    rmse_norm = rmse / truth_std[np.newaxis, :]  # (time, channel)

    # X-axis: lead time in hours
    lead_hours = np.arange(n_times) * dt_minutes / 60.0

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    for ch in channels:
        ch_name = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f'ch{ch}'
        color = CHANNEL_COLORS[ch] if ch < len(CHANNEL_COLORS) else None
        ax.plot(lead_hours, rmse_norm[:, ch], label=ch_name, linewidth=2, color=color)

    ax.set_xlabel('Lead Time (hours)', fontsize=12)
    ax.set_ylabel('Normalized RMSE (RMSE / Temporal Std Dev)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if ylim is not None:
        ax.set_ylim(ylim)

    if title is None:
        title = 'Normalized RMSE vs Lead Time'
    ax.set_title(title, fontsize=14)

    fig.tight_layout()
    return fig


# ── 3. Variance vs Lead Time ────────────────────────────────────────────────────

def plot_variance_vs_leadtime(
    preds,
    truth,
    channels=None,
    dt_minutes=10.0,
    n_times_plot=None,
    title=None,
    figsize=(10, 6),
):
    """
    Area-weighted variance vs lead time (forecast vs truth) for each variable.

    Useful for detecting variance growth (instability) or damping.

    Parameters
    ----------
    preds, truth : array-like, shape (n_samples, n_times, n_channels, nlat, nlon)
    channels : list of int, optional
        Which channels to plot. None = all.
    dt_minutes : float
        Time step in minutes between successive lead times.
    n_times_plot : int, optional
        Only plot the first N timesteps (useful for zooming into early behavior).
    title : str, optional
    figsize : tuple

    Returns
    -------
    matplotlib.Figure
    """
    preds = _to_numpy(preds)
    truth = _to_numpy(truth)

    n_samples, n_times, n_channels, nlat, nlon = preds.shape

    if channels is None:
        channels = list(range(n_channels))
    if n_times_plot is not None:
        preds = preds[:, :n_times_plot]
        truth = truth[:, :n_times_plot]
        n_times = n_times_plot

    # Area weights
    cos_w = _cos_weights(nlat)
    weights = cos_w[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]

    # Area-weighted mean per sample/time/channel: (sample, time, channel)
    pred_mean = (preds * weights).sum(axis=-2).mean(axis=-1)
    truth_mean = (truth * weights).sum(axis=-2).mean(axis=-1)

    # Area-weighted variance: E[(x - mean)^2]
    pred_var = (((preds - pred_mean[..., np.newaxis, np.newaxis]) ** 2) * weights).sum(axis=-2).mean(axis=-1)
    truth_var = (((truth - truth_mean[..., np.newaxis, np.newaxis]) ** 2) * weights).sum(axis=-2).mean(axis=-1)

    # Average over samples → (time, channel)
    pred_var_avg = pred_var.mean(axis=0)
    truth_var_avg = truth_var.mean(axis=0)

    # Normalize by time-mean truth variance per channel
    norm = truth_var_avg.mean(axis=0)  # (channel,)
    norm = np.where(norm > 0, norm, 1.0)  # avoid division by zero
    pred_var_norm = pred_var_avg / norm[np.newaxis, :]
    truth_var_norm = truth_var_avg / norm[np.newaxis, :]

    # X-axis
    lead_hours = np.arange(n_times) * dt_minutes / 60.0

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    for ch in channels:
        ch_name = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f'ch{ch}'
        color = CHANNEL_COLORS[ch] if ch < len(CHANNEL_COLORS) else None
        ax.plot(lead_hours, pred_var_norm[:, ch],
                label=f'{ch_name} (forecast)', linewidth=2,
                color=color, linestyle='-', alpha=0.8)
        ax.plot(lead_hours, truth_var_norm[:, ch],
                label=f'{ch_name} (truth)', linewidth=2,
                color=color, linestyle='--', alpha=0.6)

    ax.set_xlabel('Lead Time (hours)', fontsize=12)
    ax.set_ylabel('Normalized Variance (Var / Mean Truth Var)', fontsize=12)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    if title is None:
        title = 'Normalized Variance vs Lead Time'
    ax.set_title(title, fontsize=14)

    fig.tight_layout()
    return fig
