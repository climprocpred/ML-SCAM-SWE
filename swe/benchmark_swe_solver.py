#!/usr/bin/env python
# coding=utf-8
#%%
"""
Benchmark GPU timing for the PyTorch ShallowWaterSolver.

Measures:
  - IC generation time (random_initial_condition)
  - Solver rollout time for varying rollout lengths
  - Per-step and per-100-step averages to reveal initialization overhead
  - spec2grid / grid2spec round-trip overhead

Usage:
    conda run -n pytorch3 python benchmark_swe_solver.py [--trials 5] [--csv results.csv]
"""

import time
import argparse
import csv
import sys
from math import ceil

import torch
from torch_harmonics.examples.shallow_water_equations import ShallowWaterSolver


def sync():
    """CUDA synchronize (no-op on CPU)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_fn(fn, *args, **kwargs):
    """Time a function with CUDA sync. Returns (result, elapsed_seconds)."""
    sync()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    sync()
    elapsed = time.perf_counter() - t0
    return result, elapsed


def benchmark_ic_generation(solver, n_trials=20, mach=0.2):
    """Measure IC generation time."""
    # Warmup
    for _ in range(3):
        solver.random_initial_condition(mach=mach)
    sync()

    times = []
    for _ in range(n_trials):
        _, elapsed = time_fn(solver.random_initial_condition, mach=mach)
        times.append(elapsed)

    return times


def benchmark_spec2grid_grid2spec(solver, n_trials=20, mach=0.2):
    """Measure spec2grid and grid2spec round-trip time."""
    ic_spec = solver.random_initial_condition(mach=mach)
    sync()

    # Warmup
    for _ in range(3):
        grid = solver.spec2grid(ic_spec)
        solver.grid2spec(grid)
    sync()

    s2g_times = []
    g2s_times = []
    for _ in range(n_trials):
        _, elapsed = time_fn(solver.spec2grid, ic_spec)
        s2g_times.append(elapsed)

        grid = solver.spec2grid(ic_spec)
        sync()
        _, elapsed = time_fn(solver.grid2spec, grid)
        g2s_times.append(elapsed)

    return s2g_times, g2s_times


def benchmark_rollout(solver, rollout_steps, n_trials=5, mach=0.2):
    """
    Benchmark a single rollout length.

    Returns dict with:
      - rollout_steps: number of solver steps
      - total_times: list of total elapsed times per trial
      - ic_times: list of IC generation times per trial (excluded from rollout time)
    """
    # Warmup: do one full rollout to prime CUDA kernels
    ic = solver.random_initial_condition(mach=mach)
    solver.timestep(ic, rollout_steps)
    sync()

    total_times = []
    ic_times = []

    for _ in range(n_trials):
        # Time IC generation separately
        ic, ic_elapsed = time_fn(solver.random_initial_condition, mach=mach)
        ic_times.append(ic_elapsed)

        # Time the rollout only
        _, rollout_elapsed = time_fn(solver.timestep, ic, rollout_steps)
        total_times.append(rollout_elapsed)

    return {
        "rollout_steps": rollout_steps,
        "total_times": total_times,
        "ic_times": ic_times,
    }


def fmt_ms(seconds):
    """Format seconds as milliseconds string."""
    return f"{seconds * 1000:.3f}"


def main():
    parser = argparse.ArgumentParser(description="Benchmark SWE solver GPU timing")
    parser.add_argument("--trials", type=int, default=5,
                        help="Number of trials per rollout length (default: 5)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional CSV file to save results")
    parser.add_argument("--mach", type=float, default=0.2,
                        help="Mach number for IC generation (default: 0.2)")
    parser.add_argument(
        "--rollout-steps", type=str, default="4,8,20,40,100,200,400,1000,2000,4000",
        help="Comma-separated list of rollout lengths in solver steps (default: 4,8,20,40,100,200,400,1000,2000,4000)"
    )
    args = parser.parse_args()

    rollout_lengths = [int(x) for x in args.rollout_steps.split(",")]

    # ── Device ──────────────────────────────────────────────────────────
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device.index)
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("WARNING: No GPU detected — timings will be CPU-only")

    # ── Solver setup (matches TH_SWE.py config) ────────────────────────
    nlat, nlon = 192, 288
    dt_solver = 150  # seconds
    grid = "equiangular"
    lmax = ceil(nlat / 3)  # 64
    mmax = lmax

    print(f"\nSolver config: nlat={nlat}, nlon={nlon}, dt_solver={dt_solver}s, "
          f"lmax={lmax}, mmax={mmax}, grid={grid}")
    print(f"Mach={args.mach}, Trials per rollout length={args.trials}")

    solver = ShallowWaterSolver(
        nlat, nlon, dt_solver, lmax=lmax, mmax=mmax, grid=grid
    ).to(device).float()
    sync()
    print("Solver created and moved to device.\n")

    # ── IC generation benchmark ─────────────────────────────────────────
    print("=" * 80)
    print("IC GENERATION TIMING (random_initial_condition)")
    print("=" * 80)
    ic_times = benchmark_ic_generation(solver, n_trials=20, mach=args.mach)
    ic_mean = sum(ic_times) / len(ic_times)
    ic_min = min(ic_times)
    ic_max = max(ic_times)
    print(f"  Mean: {fmt_ms(ic_mean)} ms | Min: {fmt_ms(ic_min)} ms | Max: {fmt_ms(ic_max)} ms  (n=20)")
    print()

    # ── spec2grid / grid2spec benchmark ─────────────────────────────────
    print("=" * 80)
    print("SPEC2GRID / GRID2SPEC TIMING")
    print("=" * 80)
    s2g_times, g2s_times = benchmark_spec2grid_grid2spec(solver, n_trials=20, mach=args.mach)
    s2g_mean = sum(s2g_times) / len(s2g_times)
    g2s_mean = sum(g2s_times) / len(g2s_times)
    print(f"  spec2grid  Mean: {fmt_ms(s2g_mean)} ms  (n=20)")
    print(f"  grid2spec  Mean: {fmt_ms(g2s_mean)} ms  (n=20)")
    print()

    # ── Rollout benchmarks ──────────────────────────────────────────────
    print("=" * 80)
    print("ROLLOUT TIMING (solver.timestep)")
    print("=" * 80)
    print()

    # Config context
    dt_ml_step = 600  # seconds per ML step (as in TH_SWE.py)
    nsteps_per_ml = dt_ml_step // dt_solver  # 4 solver steps per ML step
    print(f"Note: 1 ML step = {nsteps_per_ml} solver steps = {dt_ml_step}s of simulated time")
    print(f"      100 ML steps = {100 * nsteps_per_ml} solver steps = {100 * dt_ml_step}s simulated")
    print()

    results = []
    for n_steps in rollout_lengths:
        result = benchmark_rollout(solver, n_steps, n_trials=args.trials, mach=args.mach)
        results.append(result)

    # ── Print results table ─────────────────────────────────────────────
    header = (
        f"{'Steps':>8} | {'ML Steps':>8} | {'Sim Time':>10} | "
        f"{'Total (ms)':>12} | {'Per Step (ms)':>14} | "
        f"{'Per 100 ML Steps (ms)':>22} | {'IC (ms)':>10}"
    )
    print(header)
    print("-" * len(header))

    csv_rows = []
    for r in results:
        n = r["rollout_steps"]
        total_mean = sum(r["total_times"]) / len(r["total_times"])
        ic_mean_r = sum(r["ic_times"]) / len(r["ic_times"])
        per_step = total_mean / n
        ml_steps_equiv = n / nsteps_per_ml
        per_100_ml = (total_mean / ml_steps_equiv) * 100 if ml_steps_equiv > 0 else 0
        sim_time_s = n * dt_solver  # simulated time in seconds

        # Format simulated time nicely
        if sim_time_s >= 3600:
            sim_str = f"{sim_time_s/3600:.1f}h"
        elif sim_time_s >= 60:
            sim_str = f"{sim_time_s/60:.1f}m"
        else:
            sim_str = f"{sim_time_s:.0f}s"

        row_str = (
            f"{n:>8} | {ml_steps_equiv:>8.1f} | {sim_str:>10} | "
            f"{fmt_ms(total_mean):>12} | {fmt_ms(per_step):>14} | "
            f"{fmt_ms(per_100_ml):>22} | {fmt_ms(ic_mean_r):>10}"
        )
        print(row_str)

        csv_rows.append({
            "solver_steps": n,
            "ml_steps": ml_steps_equiv,
            "simulated_time_s": sim_time_s,
            "total_ms": total_mean * 1000,
            "per_step_ms": per_step * 1000,
            "per_100_ml_steps_ms": per_100_ml * 1000,
            "ic_gen_ms": ic_mean_r * 1000,
        })

    print()

    # ── Interpretation ──────────────────────────────────────────────────
    if len(results) >= 2:
        short = results[0]
        long = results[-1]
        short_per_step = (sum(short["total_times"]) / len(short["total_times"])) / short["rollout_steps"]
        long_per_step = (sum(long["total_times"]) / len(long["total_times"])) / long["rollout_steps"]

        ratio = short_per_step / long_per_step if long_per_step > 0 else float('inf')
        print(f"Overhead ratio (shortest vs longest per-step time): {ratio:.2f}x")
        if ratio > 1.2:
            print(f"  → Significant per-rollout overhead detected: short rollouts are ~{ratio:.1f}x slower per step.")
            print(f"     This means longer rollouts amortize startup cost better.")
        elif ratio > 1.05:
            print(f"  → Modest per-rollout overhead (~{(ratio-1)*100:.0f}% slower for short rollouts).")
        else:
            print(f"  → Negligible per-rollout overhead. Per-step cost is nearly constant.")
    print()

    # ── First-step vs subsequent-steps analysis ─────────────────────────
    print("=" * 80)
    print("FIRST-STEP VS SUBSEQUENT-STEPS ANALYSIS")
    print("=" * 80)
    print("(Measures time for solver.timestep(spec, 1) followed by solver.timestep(spec, 1) × 9)")
    print()

    ic = solver.random_initial_condition(mach=args.mach)
    # Warmup
    solver.timestep(ic, 1)
    sync()

    first_step_times = []
    subsequent_step_times = []
    n_analysis_trials = 10

    for _ in range(n_analysis_trials):
        # Fresh IC each trial
        ic = solver.random_initial_condition(mach=args.mach)
        sync()

        # First step
        result, t_first = time_fn(solver.timestep, ic, 1)
        first_step_times.append(t_first)

        # Next 9 steps (one at a time)
        spec = result
        for _ in range(9):
            spec, t_sub = time_fn(solver.timestep, spec, 1)
            subsequent_step_times.append(t_sub)

    first_mean = sum(first_step_times) / len(first_step_times)
    subseq_mean = sum(subsequent_step_times) / len(subsequent_step_times)
    print(f"  First step after new IC:   {fmt_ms(first_mean)} ms  (n={len(first_step_times)})")
    print(f"  Subsequent steps (2-10):   {fmt_ms(subseq_mean)} ms  (n={len(subsequent_step_times)})")
    print(f"  Ratio (first / subsequent): {first_mean/subseq_mean:.2f}x")

    # Note about Adams-Bashforth startup
    print()
    print("  Note: The solver uses Adams-Bashforth (3rd order). The first 2 steps use")
    print("  forward Euler / 2nd-order AB as startup, which is computationally identical")
    print("  to later steps (same kernel, different coefficients). Any difference is")
    print("  likely due to GPU memory allocation on the very first call.")
    print()

    # ── Optional CSV output ─────────────────────────────────────────────
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Results saved to {args.csv}")


if __name__ == "__main__":
    main()

# %%
