"""
Pre-computed trajectory dataset for SWE training.

Loads solver trajectories from a Zarr database and serves contiguous time windows
as training samples. Supports multi-history inputs and multi-step targets.

The flat index design ensures:
  - Every sample is a contiguous time slice from a single trajectory
  - Standard DataLoader(shuffle=True) randomizes across all trajectories and time offsets
  - n_history and nfuture are handled uniformly via window size
"""

import os



import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import zarr
except ImportError:
    raise ImportError("zarr is required for PrecomputedTrajectoryDataset. Install with: pip install zarr")


import time # Added for retry logic

class PrecomputedTrajectoryDataset(Dataset):
    """
    Dataset that loads contiguous time windows from pre-computed Zarr trajectories.
    
    Each sample is a window of `n_history + nfuture + 1` consecutive frames from
    a single trajectory. The first `n_history` frames form the input (stacked along
    the channel dimension), and the remaining `nfuture + 1` frames are the target.
    
    Parameters
    ----------
    zarr_path : str
        Path to the Zarr store containing 'states' dataset with shape (N, T, C, H, W)
    stats_path : str
        Path to .npz file containing 'inp_mean' and 'inp_var' arrays
    n_history : int
        Number of past timesteps to include in input (1 = single-step)
    nfuture : int
        Number of future timesteps beyond the immediate next step (0 = single target)
    normalize : bool
        Whether to apply normalization using the stored statistics
    """

    def __init__(self, zarr_path, stats_path, n_history=1, nfuture=0, normalize=True):
        super().__init__()

        self.n_history = n_history
        self.nfuture = nfuture
        self.normalize = normalize
        self.window = n_history + nfuture + 1  # total frames per sample

        # Open Zarr store (lazy, no data loaded yet)
        # Using mode='r' usually implies a read-only store.
        self.store = zarr.open(zarr_path, mode='r')
        self.states = self.store['states']  # (N, T+1, C, H, W)
        self.N, self.T_total, self.C, self.H, self.W = self.states.shape

        if self.window > self.T_total:
            raise ValueError(
                f"Window size ({self.window} = {n_history} history + {nfuture} future + 1) "
                f"exceeds trajectory length ({self.T_total})"
            )

        # Load normalization stats
        stats = np.load(stats_path)
        self.inp_mean = torch.from_numpy(stats['inp_mean']).float()  # (C, 1, 1)
        self.inp_var = torch.from_numpy(stats['inp_var']).float()    # (C, 1, 1)

        # Build flat index: list of (trajectory_idx, t_start)
        # Each entry represents one valid training sample
        self.index = []
        n_windows_per_traj = self.T_total - self.window + 1
        for i in range(self.N):
            for t in range(n_windows_per_traj):
                self.index.append((i, t))

        print(f"PrecomputedTrajectoryDataset: {self.N} trajectories, "
              f"{self.T_total} steps each, window={self.window}")
        print(f"  {len(self.index)} total samples ({n_windows_per_traj} per trajectory)")
        print(f"  n_history={n_history}, nfuture={nfuture}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        traj_i, t_start = self.index[idx]

        # Read contiguous time window from Zarr (efficient chunked read)
        # Retry logic for transient filesystem errors (e.g. OSError: [Errno 5] Input/output error)
        max_retries = 10
        chunk = None
        for attempt in range(max_retries):
            try:
                chunk = self.states[traj_i, t_start: t_start + self.window]
                break
            except (OSError, IOError) as e:
                if attempt < max_retries - 1:
                    sleep_time = 0.1 * (2 ** attempt)  # exponential backoff
                    print(f"Zarr read error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                else:
                    raise e
        
        chunk = torch.from_numpy(np.asarray(chunk)).float()  # (window, C, H, W)

        # Split into input history and target(s)
        inp_frames = chunk[:self.n_history]    # (n_history, C, H, W)
        tar_frames = chunk[self.n_history:]    # (nfuture+1, C, H, W)

        # Flatten history into channel dim: (n_history*C, H, W)
        inp = inp_frames.reshape(-1, self.H, self.W)

        # Normalize input
        if self.normalize:
            # Repeat stats for multi-history stacking
            mean_rep = self.inp_mean.repeat(self.n_history, 1, 1)  # (n_history*C, 1, 1)
            var_rep = self.inp_var.repeat(self.n_history, 1, 1)
            inp = (inp - mean_rep) / torch.sqrt(var_rep)

        # Target: single or trajectory
        if self.nfuture > 0:
            tar = tar_frames  # (nfuture+1, C, H, W)
            if self.normalize:
                tar = (tar - self.inp_mean) / torch.sqrt(self.inp_var)
        else:
            tar = tar_frames[0]  # (C, H, W)
            if self.normalize:
                tar = (tar - self.inp_mean.squeeze(0)) / torch.sqrt(self.inp_var.squeeze(0))

        return inp, tar

    def set_num_examples(self, n):
        """No-op for compatibility with PdeDatasetExtended interface."""
        pass

    def set_initial_condition(self, ic_type):
        """No-op for compatibility with PdeDatasetExtended interface."""
        pass
