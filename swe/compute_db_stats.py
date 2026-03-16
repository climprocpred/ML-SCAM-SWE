#!/usr/bin/env python
"""
Compute normalization statistics from an existing Zarr trajectory database.
H/vort/div stats (inp_mean, inp_var) are computed in pure numpy — no GPU needed.
UV stats (uv_mean, uv_var) require the solver; use --compute_uv to compute them
on CPU, or they will be loaded from any existing stats.npz if already present.

Usage:
    # H/vort/div stats only (fast, CPU, no dependencies):
    python compute_db_stats.py --db_dir /glade/derecho/scratch/idavis/swe_training_db

    # H/vort/div + UV stats (needs torch + pytorch3 env, slower):
    python compute_db_stats.py --db_dir /glade/derecho/scratch/idavis/swe_training_db --compute_uv

    # Use all frames instead of random sample:
    python compute_db_stats.py --db_dir /glade/derecho/scratch/idavis/swe_training_db --compute_uv --all_frames
"""
#%%
import os
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import zarr
except ImportError:
    print("ERROR: zarr not installed. Run: pip install zarr")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x


def compute_stats(args):
    print(f"Starting compute_db_stats...", flush=True)
    zarr_path = os.path.join(args.db_dir, "trajectories.zarr")
    stats_path = os.path.join(args.db_dir, "stats.npz")

    if not os.path.exists(zarr_path):
        print(f"ERROR: Zarr store not found at {zarr_path}")
        sys.exit(1)

    print(f"Opening Zarr store: {zarr_path}", flush=True)
    store = zarr.open(zarr_path, mode='r')
    states = store['states']          # (N, T+1, C, H, W)
    N, T_total, C, H, W = states.shape
    total_frames = N * T_total

    print(f"Database: {zarr_path}", flush=True)
    print(f"  Shape: {states.shape}  ({N} trajectories × {T_total} steps, {C} channels, {H}×{W})", flush=True)
    print(f"  Total frames: {total_frames:,}", flush=True)

    if args.all_frames:
        # Iterate every frame in trajectory order (sequential = fast Zarr reads)
        print(f"\nMode: ALL {total_frames:,} frames (sequential)")
        n_samples = total_frames
        use_random = False
    else:
        n_samples = min(args.n_samples, total_frames)
        print(f"\nMode: {n_samples:,} randomly sampled frames")
        use_random = True

    num_workers = args.num_workers
    print(f"  Workers: {num_workers}", flush=True)

    # ── Compute mean and variance ─────────────────────────────────────────
    # Single-pass: work in float32 to minimize memory bandwidth, accumulate in float64.
    sum_mean = np.zeros(C, dtype=np.float64)
    sum_sq   = np.zeros(C, dtype=np.float64)
    count = 0
    lock = threading.Lock()

    start = time.time()

    if use_random:
        rng = np.random.default_rng(args.seed)
        traj_idx = rng.integers(0, N, size=n_samples)
        time_idx = rng.integers(0, T_total, size=n_samples)

        # Sort by trajectory then time for better cache locality
        sort_order = np.lexsort((time_idx, traj_idx))
        traj_idx = traj_idx[sort_order]
        time_idx = time_idx[sort_order]

        def read_frame(i):
            frame = np.asarray(states[int(traj_idx[i]), int(time_idx[i])], dtype=np.float32)
            return frame.mean(axis=(-2, -1)).astype(np.float64), frame.var(axis=(-2, -1)).astype(np.float64)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(read_frame, i): i for i in range(n_samples)}
            for fut in tqdm(as_completed(futures), total=n_samples, desc="Computing stats"):
                m, v = fut.result()
                with lock:
                    sum_mean += m
                    sum_sq   += v
                    count    += 1

    else:
        # All frames: each worker loads one full trajectory
        def process_traj(ti):
            traj_data = np.asarray(states[ti], dtype=np.float32)  # (T+1, C, H, W)
            frame_means = traj_data.mean(axis=(-2, -1))            # (T+1, C)
            frame_vars  = traj_data.var(axis=(-2, -1))              # (T+1, C)
            return (frame_means.sum(axis=0).astype(np.float64),
                    frame_vars.sum(axis=0).astype(np.float64),
                    traj_data.shape[0])

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_traj, ti): ti for ti in range(N)}
            for fut in tqdm(as_completed(futures), total=N, desc="Trajectories"):
                ms, vs, n = fut.result()
                with lock:
                    sum_mean += ms
                    sum_sq   += vs
                    count    += n

    mean = sum_mean / count
    var  = sum_sq / count

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    mean_f32 = mean.astype(np.float32).reshape(C, 1, 1)
    var_f32  = var.astype(np.float32).reshape(C, 1, 1)

    channel_names = ["height", "vorticity", "divergence"]
    print("\nNormalization statistics:")
    print(f"  {'Channel':<12}  {'Mean':>12}  {'Variance':>12}  {'Std':>12}")
    print(f"  {'-'*52}")
    for c in range(C):
        print(f"  {channel_names[c]:<12}  {mean[c]:>12.4e}  {var[c]:>12.4e}  {np.sqrt(var[c]):>12.4e}")

    # ── UV stats ─────────────────────────────────────────────────────────
    uv_mean_out = np.zeros((2, 1, 1), dtype=np.float32)
    uv_var_out  = np.ones((2, 1, 1), dtype=np.float32)

    if args.compute_uv:
        print("\nComputing UV stats using CPU solver (this may take a few minutes)...")
        try:
            import torch
            from pde_dataset_extended import PdeDatasetExtended
        except ImportError as e:
            print(f"ERROR: --compute_uv requires torch and pde_dataset_extended: {e}")
            sys.exit(1)

        # Read solver config from Zarr metadata
        attrs = dict(store.attrs)
        dt        = int(attrs.get('dt', 600))
        dt_solver = int(attrs.get('dt_solver', 150))
        nsteps    = dt // dt_solver
        grid      = attrs.get('grid', 'equiangular')

        print(f"  Solver config from DB metadata: dt={dt}, dt_solver={dt_solver}, grid={grid}, nlat={H}, nlon={W}")

        # Instantiate solver on CPU (stable enough for inference/transforms)
        device = torch.device('cpu')
        uv_dataset = PdeDatasetExtended(
            dt=dt, nsteps=nsteps, dims=(H, W), device=device, grid=grid,
            normalize=False, nfuture=0, num_examples=1,
        )
        solver = uv_dataset.solver

        uv_n = min(args.n_uv_samples, N * T_total)
        rng_uv = np.random.default_rng(args.seed)
        uv_traj_idx = rng_uv.integers(0, N, size=uv_n)
        uv_time_idx = rng_uv.integers(0, T_total, size=uv_n)
        # Sort for cache locality
        sort_order = np.lexsort((uv_time_idx, uv_traj_idx))
        uv_traj_idx = uv_traj_idx[sort_order]
        uv_time_idx = uv_time_idx[sort_order]

        uv_mean_acc = np.zeros(2, dtype=np.float64)
        uv_var_acc  = np.zeros(2, dtype=np.float64)

        uv_start = time.time()
        for i in tqdm(range(uv_n), desc="UV stats"):
            ti, si = int(uv_traj_idx[i]), int(uv_time_idx[i])
            frame = torch.from_numpy(np.asarray(states[ti, si])).float()  # (C, H, W)
            frame_spec = solver.grid2spec(frame)
            uv = solver.getuv(frame_spec[1:])   # (2, H, W)
            uv_mean_acc += uv.mean(dim=(-1, -2)).numpy().astype(np.float64)
            uv_var_acc  += uv.var(dim=(-1, -2)).numpy().astype(np.float64)

        uv_mean_out = (uv_mean_acc / uv_n).astype(np.float32).reshape(2, 1, 1)
        uv_var_out  = (uv_var_acc  / uv_n).astype(np.float32).reshape(2, 1, 1)
        uv_stds = np.sqrt(uv_var_out.squeeze())
        print(f"  UV stats computed in {time.time()-uv_start:.1f}s from {uv_n} frames")
        print(f"  U_std={uv_stds[0]:.4f}, V_std={uv_stds[1]:.4f} m/s")

    elif os.path.exists(stats_path):
        # Try to preserve UV stats from an existing stats.npz
        existing = np.load(stats_path)
        if "uv_var" in existing and not np.allclose(existing["uv_var"], 0.0):
            uv_mean_out = existing["uv_mean"]
            uv_var_out  = existing["uv_var"]
            uv_stds = np.sqrt(uv_var_out.squeeze())
            print(f"\nPreserving UV stats from existing stats.npz: U_std={uv_stds[0]:.4f}, V_std={uv_stds[1]:.4f} m/s")
        else:
            print("\nWARNING: No valid UV stats found. Re-run with --compute_uv to compute them.")
    else:
        print("\nWARNING: No existing stats.npz and --compute_uv not set. UV stats will be placeholders.")
        print("  Re-run with --compute_uv to compute UV stats.")

    # ── Save ──────────────────────────────────────────────────────────────
    np.savez(
        stats_path,
        inp_mean=mean_f32,
        inp_var=var_f32,
        uv_mean=uv_mean_out,
        uv_var=uv_var_out,
    )
    print(f"\nSaved to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute normalization stats from a Zarr trajectory database"
    )
    parser.add_argument("--db_dir", type=str, required=True,
                        help="Path to the database directory (contains trajectories.zarr)")
    parser.add_argument("--n_samples", type=int, default=20000,
                        help="Number of random frames for H/vort/div stats (default: 20000)")
    parser.add_argument("--all_frames", action="store_true",
                        help="Use every frame instead of random sampling (slower but exact)")
    parser.add_argument("--compute_uv", action="store_true",
                        help="Compute UV (wind) stats using the SWE solver on CPU. "
                             "Requires pytorch3 environment. Slower but necessary for UV loss.")
    parser.add_argument("--n_uv_samples", type=int, default=500,
                        help="Number of frames for UV stats (default: 500; ignored without --compute_uv)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for frame sampling")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel reader threads (default: 8)")

    args = parser.parse_args()
    compute_stats(args)


if __name__ == "__main__":
    main()

# %%
