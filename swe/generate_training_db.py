#!/usr/bin/env python
"""
Generate a pre-computed training database of SWE solver trajectories.

Runs the SWE solver for N trajectories of T steps each, saving raw (unnormalized)
grid-space states to a Zarr store.

Stats are computed separately using compute_db_stats.py (CPU only, no GPU needed).
Use --skip_stats to skip stats computation here (recommended for GPU jobs).

Usage:
    python generate_training_db.py \
        --output_dir /glade/derecho/scratch/$USER/swe_training_db \
        --n_trajectories 500 \
        --trajectory_length 1000 \
        --stochastic_ic_mach \
        --stochastic_ic_llimit \
        --skip_stats

    # Then on a login/CPU node:
    python compute_db_stats.py --db_dir /glade/derecho/scratch/$USER/swe_training_db
"""

import os
import sys
import time
import argparse

import torch
import numpy as np

try:
    import zarr
except ImportError:
    print("ERROR: zarr is required. Install with: pip install zarr")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pde_dataset_extended import PdeDatasetExtended


def generate_trajectories(args):
    """Generate solver trajectories and save to Zarr."""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device.index)
    print(f"Using device: {device}")

    # Match solver config from TH_SWE.py
    dt = args.dt
    dt_solver = args.dt_solver
    nsteps = dt // dt_solver
    grid = args.grid
    nlat, nlon = args.nlat, args.nlon

    print(f"Solver config: dt={dt}, dt_solver={dt_solver}, nsteps={nsteps}")
    print(f"Grid: {grid}, nlat={nlat}, nlon={nlon}")

    # Create dataset (only used for solver + IC generation)
    dataset = PdeDatasetExtended(
        dt=dt, nsteps=nsteps, dims=(nlat, nlon), device=device, grid=grid,
        normalize=False,  # We store raw values
        nfuture=0,
        ic_mach=args.ic_mach,
        ic_llimit=args.ic_llimit,
        stochastic_ic_llimit=args.stochastic_ic_llimit,
        stochastic_ic_mach=args.stochastic_ic_mach,
        ic_spinup_max=args.ic_spinup_max,
        num_examples=1,
    )
    solver = dataset.solver

    N = args.n_trajectories
    T = args.trajectory_length
    C = 3  # height, vorticity, divergence

    print(f"\nGenerating {N} trajectories of {T} steps each")
    print(f"IC config: mach={args.ic_mach}, stochastic_mach={args.stochastic_ic_mach}, "
          f"stochastic_llimit={args.stochastic_ic_llimit}, spinup_max={args.ic_spinup_max}")

    # Estimate storage
    frame_bytes = C * nlat * nlon * 4  # float32
    total_bytes = N * (T + 1) * frame_bytes
    print(f"Estimated storage: {total_bytes / 1e9:.1f} GB")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create Zarr store
    zarr_path = os.path.join(args.output_dir, "trajectories.zarr")
    store = zarr.open(zarr_path, mode='w')

    # Create dataset with chunking optimized for time-window reads
    # Chunk along time axis: read 32 contiguous frames efficiently
    chunk_time = min(32, T + 1)
    states = store.create_dataset(
        'states',
        shape=(N, T + 1, C, nlat, nlon),
        chunks=(1, chunk_time, C, nlat, nlon),
        dtype='float32',
    )

    # Store metadata as attributes
    store.attrs['dt'] = dt
    store.attrs['dt_solver'] = dt_solver
    store.attrs['nsteps'] = nsteps
    store.attrs['nlat'] = nlat
    store.attrs['nlon'] = nlon
    store.attrs['grid'] = grid
    store.attrs['ic_mach'] = args.ic_mach
    store.attrs['ic_llimit'] = args.ic_llimit
    store.attrs['stochastic_ic_mach'] = args.stochastic_ic_mach
    store.attrs['stochastic_ic_llimit'] = args.stochastic_ic_llimit
    store.attrs['ic_spinup_max'] = args.ic_spinup_max
    store.attrs['n_trajectories'] = N
    store.attrs['trajectory_length'] = T

    # ── Generate trajectories ──────────────────────────────────────────
    nan_trajectories = 0
    gen_start = time.time()

    for traj_idx in tqdm(range(N), desc="Generating trajectories"):
        traj_start = time.time()

        # Generate IC using the same logic as PdeDatasetExtended
        ic_spec = dataset._get_initial_spec()

        # Allocate buffer for this trajectory on GPU
        traj_buffer = torch.empty(T + 1, C, nlat, nlon, device=device, dtype=torch.float32)

        # Store initial state (frame 0)
        traj_buffer[0] = solver.spec2grid(ic_spec)

        # Roll out solver for T steps
        current_spec = ic_spec
        is_valid = True

        for step in range(T):
            current_spec = solver.timestep(current_spec, nsteps)

            if torch.isnan(current_spec).any():
                print(f"\nWARNING: NaN at trajectory {traj_idx}, step {step+1}/{T}. Regenerating...")
                nan_trajectories += 1
                is_valid = False
                break

            traj_buffer[step + 1] = solver.spec2grid(current_spec)

        if not is_valid:
            # Retry with a new IC
            retry_success = False
            for retry in range(5):
                ic_spec = dataset._get_initial_spec()
                traj_buffer[0] = solver.spec2grid(ic_spec)
                current_spec = ic_spec
                retry_valid = True

                for step in range(T):
                    current_spec = solver.timestep(current_spec, nsteps)
                    if torch.isnan(current_spec).any():
                        retry_valid = False
                        break
                    traj_buffer[step + 1] = solver.spec2grid(current_spec)

                if retry_valid:
                    retry_success = True
                    break

            if not retry_success:
                print(f"\nERROR: Trajectory {traj_idx} failed after 5 retries. Filling with zeros.")
                traj_buffer.zero_()

        # Write to Zarr (GPU → CPU → disk)
        states[traj_idx] = traj_buffer.cpu().numpy()

        if (traj_idx + 1) % 50 == 0:
            elapsed = time.time() - gen_start
            rate = (traj_idx + 1) / elapsed
            remaining = (N - traj_idx - 1) / rate
            print(f"\n  [{traj_idx+1}/{N}] {rate:.1f} traj/s, "
                  f"~{remaining/60:.0f} min remaining")

    gen_time = time.time() - gen_start
    print(f"\nGeneration complete: {N} trajectories in {gen_time:.1f}s "
          f"({N/gen_time:.1f} traj/s)")
    if nan_trajectories > 0:
        print(f"  {nan_trajectories} trajectories required regeneration due to NaN")

    # ── Normalization statistics ────────────────────────────────────────
    # UV stats ALWAYS computed here (fast, ~500 frames, needs GPU solver which is already alive).
    # Full H/vort/div stats are expensive and can be deferred to compute_db_stats.py (CPU).
    stats_path = os.path.join(args.output_dir, "stats.npz")

    print("\nComputing UV normalization stats (fast, ~500 frames)...")
    uv_n_samples = min(500, N * (T + 1))
    uv_traj_idx = np.random.randint(0, N, size=uv_n_samples)
    uv_time_idx = np.random.randint(0, T + 1, size=uv_n_samples)
    uv_mean_acc = torch.zeros(2, device='cpu', dtype=torch.float64)
    uv_var_acc  = torch.zeros(2, device='cpu', dtype=torch.float64)

    for i in tqdm(range(uv_n_samples), desc="UV stats"):
        ti, si = int(uv_traj_idx[i]), int(uv_time_idx[i])
        frame = torch.from_numpy(np.array(states[ti, si])).to(device)
        frame_spec = solver.grid2spec(frame)
        uv = solver.getuv(frame_spec[1:])  # (2, H, W)
        uv_mean_acc += uv.float().mean(dim=(-1, -2)).cpu().double()
        uv_var_acc  += uv.float().var(dim=(-1, -2)).cpu().double()

    uv_mean_final = (uv_mean_acc / uv_n_samples).float().reshape(2, 1, 1)
    uv_var_final  = (uv_var_acc  / uv_n_samples).float().reshape(2, 1, 1)
    print(f"  UV mean: {uv_mean_final.squeeze().numpy()}")
    print(f"  UV std:  {uv_var_final.squeeze().sqrt().numpy()}")

    if args.skip_stats:
        # Save UV stats with placeholder H/vort/div stats.
        # compute_db_stats.py will fill in the real inp_mean/inp_var later.
        print("\nSkipping full H/vort/div stats (--skip_stats).")
        print(f"  Run after job: python compute_db_stats.py --db_dir {args.output_dir}")
        np.savez(
            stats_path,
            inp_mean=np.zeros((C, 1, 1), dtype=np.float32),   # placeholder
            inp_var=np.ones((C, 1, 1), dtype=np.float32),     # placeholder
            uv_mean=uv_mean_final.numpy(),
            uv_var=uv_var_final.numpy(),
        )
    else:
        print("\nComputing full normalization stats (H/vort/div)...")
        stats_start = time.time()

        n_stat_samples = min(args.n_stat_samples, N * (T + 1))
        count = 0
        mean_acc = torch.zeros(C, device='cpu', dtype=torch.float64)
        var_acc  = torch.zeros(C, device='cpu', dtype=torch.float64)

        sample_traj_indices = np.random.randint(0, N, size=n_stat_samples)
        sample_time_indices = np.random.randint(0, T + 1, size=n_stat_samples)

        for i in tqdm(range(n_stat_samples), desc="H/vort/div stats"):
            ti, si = int(sample_traj_indices[i]), int(sample_time_indices[i])
            frame = torch.from_numpy(np.array(states[ti, si])).to(device)
            mean_acc += frame.float().mean(dim=(-1, -2)).cpu().double()
            var_acc  += frame.float().var(dim=(-1, -2)).cpu().double()
            count += 1

        mean_final = (mean_acc / count).float()
        var_final  = (var_acc  / count).float()

        np.savez(
            stats_path,
            inp_mean=mean_final.reshape(-1, 1, 1).numpy(),
            inp_var=var_final.reshape(-1, 1, 1).numpy(),
            uv_mean=uv_mean_final.numpy(),
            uv_var=uv_var_final.numpy(),
        )

        stats_time = time.time() - stats_start
        print(f"Full stats in {stats_time:.1f}s from {count} samples")
        print(f"  State mean: {mean_final.numpy()}")
        print(f"  State var:  {var_final.numpy()}")

    # ── Summary ─────────────────────────────────────────────────────────
    total_time = time.time() - gen_start
    actual_size_gb = os.path.getsize(zarr_path) / 1e9 if os.path.isfile(zarr_path) else -1

    print(f"\n{'='*60}")
    print(f"Database saved to: {args.output_dir}")
    print(f"  Zarr:  {zarr_path}")
    print(f"  Stats: {stats_path}")
    print(f"  Shape: ({N}, {T+1}, {C}, {nlat}, {nlon})")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate pre-computed SWE training trajectories"
    )

    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the Zarr database and stats")

    # Trajectory config
    parser.add_argument("--n_trajectories", type=int, default=500,
                        help="Number of independent trajectories to generate")
    parser.add_argument("--trajectory_length", type=int, default=1000,
                        help="Number of solver macro-steps per trajectory")

    # Solver config (defaults match TH_SWE.py)
    parser.add_argument("--dt", type=int, default=600,
                        help="Output timestep in seconds")
    parser.add_argument("--dt_solver", type=int, default=150,
                        help="Internal solver timestep in seconds")
    parser.add_argument("--nlat", type=int, default=192)
    parser.add_argument("--nlon", type=int, default=288)
    parser.add_argument("--grid", type=str, default="equiangular")

    # IC config (defaults match TH_SWE.py)
    parser.add_argument("--ic_mach", type=float, default=0.2)
    parser.add_argument("--ic_llimit", type=int, default=25)
    parser.add_argument("--stochastic_ic_mach", action="store_true")
    parser.add_argument("--stochastic_ic_llimit", action="store_true")
    parser.add_argument("--ic_spinup_max", type=int, default=0)

    # Stats config
    parser.add_argument("--skip_stats", action="store_true",
                        help="Skip stats computation (run compute_db_stats.py separately on CPU)")
    parser.add_argument("--n_stat_samples", type=int, default=5000,
                        help="Number of random frames to sample for normalization stats (ignored if --skip_stats)")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Using seed: {args.seed}")

    generate_trajectories(args)


if __name__ == "__main__":
    main()
