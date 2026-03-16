#!/usr/bin/env python
"""
Test script for PrecomputedTrajectoryDataset.

Creates a small synthetic Zarr database and verifies:
  - Correct index building for various n_history/nfuture combos
  - Correct output shapes from __getitem__
  - DataLoader batching works
  - Normalization is applied correctly

Can be run on login node (no GPU needed).
"""

import os
import sys
import tempfile
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import zarr
except ImportError:
    print("ERROR: zarr is required. Install with: pip install zarr")
    sys.exit(1)

from precomputed_dataset import PrecomputedTrajectoryDataset


def create_test_db(tmpdir, N=3, T=10, C=3, H=8, W=16):
    """Create a small synthetic Zarr database for testing."""
    zarr_path = os.path.join(tmpdir, "trajectories.zarr")
    stats_path = os.path.join(tmpdir, "stats.npz")

    store = zarr.open(zarr_path, mode='w')
    states = store.create_dataset(
        'states',
        shape=(N, T + 1, C, H, W),
        chunks=(1, T + 1, C, H, W),
        dtype='float32',
    )

    # Fill with known values: state[traj, t, c, h, w] = traj*1000 + t*10 + c
    for i in range(N):
        for t in range(T + 1):
            for c in range(C):
                states[i, t, c] = i * 1000 + t * 10 + c

    # Create simple stats
    np.savez(
        stats_path,
        inp_mean=np.zeros((C, 1, 1), dtype=np.float32),
        inp_var=np.ones((C, 1, 1), dtype=np.float32),
    )

    return zarr_path, stats_path, N, T, C, H, W


def test_single_step():
    """Test n_history=1, nfuture=0 (basic single-step)."""
    print("=" * 60)
    print("TEST: Single-step (n_history=1, nfuture=0)")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp()
    try:
        zarr_path, stats_path, N, T, C, H, W = create_test_db(tmpdir)

        ds = PrecomputedTrajectoryDataset(
            zarr_path=zarr_path, stats_path=stats_path,
            n_history=1, nfuture=0, normalize=False,
        )

        # Window = 1 + 0 + 1 = 2
        # Samples per traj = 11 - 2 + 1 = 10
        expected_len = N * (T + 1 - 2 + 1)
        assert len(ds) == expected_len, f"Expected {expected_len}, got {len(ds)}"
        print(f"  ✓ Length: {len(ds)} (expected {expected_len})")

        inp, tar = ds[0]
        assert inp.shape == (C, H, W), f"Expected inp ({C},{H},{W}), got {inp.shape}"
        assert tar.shape == (C, H, W), f"Expected tar ({C},{H},{W}), got {tar.shape}"
        print(f"  ✓ Shapes: inp={inp.shape}, tar={tar.shape}")

        # Check values: first sample is traj=0, t_start=0
        # inp = frame 0, tar = frame 1
        assert inp[0, 0, 0].item() == 0.0, f"Expected inp value 0, got {inp[0,0,0].item()}"
        assert tar[0, 0, 0].item() == 10.0, f"Expected tar value 10, got {tar[0,0,0].item()}"
        print(f"  ✓ Values correct (inp[0]={inp[0,0,0].item()}, tar[0]={tar[0,0,0].item()})")

        print("  PASSED\n")
    finally:
        shutil.rmtree(tmpdir)


def test_multi_history():
    """Test n_history=3, nfuture=0."""
    print("=" * 60)
    print("TEST: Multi-history (n_history=3, nfuture=0)")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp()
    try:
        zarr_path, stats_path, N, T, C, H, W = create_test_db(tmpdir)

        ds = PrecomputedTrajectoryDataset(
            zarr_path=zarr_path, stats_path=stats_path,
            n_history=3, nfuture=0, normalize=False,
        )

        # Window = 3 + 0 + 1 = 4
        # Samples per traj = 11 - 4 + 1 = 8
        expected_len = N * (T + 1 - 4 + 1)
        assert len(ds) == expected_len, f"Expected {expected_len}, got {len(ds)}"
        print(f"  ✓ Length: {len(ds)} (expected {expected_len})")

        inp, tar = ds[0]
        assert inp.shape == (3 * C, H, W), f"Expected inp ({3*C},{H},{W}), got {inp.shape}"
        assert tar.shape == (C, H, W), f"Expected tar ({C},{H},{W}), got {tar.shape}"
        print(f"  ✓ Shapes: inp={inp.shape}, tar={tar.shape}")

        # Check: inp channels 0..2 = frame 0, channels 3..5 = frame 1, channels 6..8 = frame 2
        # tar = frame 3
        assert inp[0, 0, 0].item() == 0.0   # traj 0, t=0, c=0
        assert inp[3, 0, 0].item() == 10.0  # traj 0, t=1, c=0
        assert inp[6, 0, 0].item() == 20.0  # traj 0, t=2, c=0
        assert tar[0, 0, 0].item() == 30.0  # traj 0, t=3, c=0
        print(f"  ✓ History stacking correct")

        print("  PASSED\n")
    finally:
        shutil.rmtree(tmpdir)


def test_multi_future():
    """Test n_history=1, nfuture=4."""
    print("=" * 60)
    print("TEST: Multi-future (n_history=1, nfuture=4)")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp()
    try:
        zarr_path, stats_path, N, T, C, H, W = create_test_db(tmpdir)

        ds = PrecomputedTrajectoryDataset(
            zarr_path=zarr_path, stats_path=stats_path,
            n_history=1, nfuture=4, normalize=False,
        )

        # Window = 1 + 4 + 1 = 6
        # Samples per traj = 11 - 6 + 1 = 6
        expected_len = N * (T + 1 - 6 + 1)
        assert len(ds) == expected_len, f"Expected {expected_len}, got {len(ds)}"
        print(f"  ✓ Length: {len(ds)} (expected {expected_len})")

        inp, tar = ds[0]
        assert inp.shape == (C, H, W), f"Expected inp ({C},{H},{W}), got {inp.shape}"
        assert tar.shape == (5, C, H, W), f"Expected tar (5,{C},{H},{W}), got {tar.shape}"
        print(f"  ✓ Shapes: inp={inp.shape}, tar={tar.shape}")

        # Check: inp = frame 0, tar = frames 1..5
        assert tar[0, 0, 0, 0].item() == 10.0   # t=1
        assert tar[4, 0, 0, 0].item() == 50.0   # t=5
        print(f"  ✓ Target trajectory values correct")

        print("  PASSED\n")
    finally:
        shutil.rmtree(tmpdir)


def test_normalization():
    """Test that normalization is applied correctly."""
    print("=" * 60)
    print("TEST: Normalization")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp()
    try:
        zarr_path, stats_path, N, T, C, H, W = create_test_db(tmpdir)

        # Override stats with non-trivial values
        np.savez(
            stats_path,
            inp_mean=np.full((C, 1, 1), 5.0, dtype=np.float32),
            inp_var=np.full((C, 1, 1), 4.0, dtype=np.float32),
        )

        ds_norm = PrecomputedTrajectoryDataset(
            zarr_path=zarr_path, stats_path=stats_path,
            n_history=1, nfuture=0, normalize=True,
        )
        ds_raw = PrecomputedTrajectoryDataset(
            zarr_path=zarr_path, stats_path=stats_path,
            n_history=1, nfuture=0, normalize=False,
        )

        inp_norm, tar_norm = ds_norm[0]
        inp_raw, tar_raw = ds_raw[0]

        # Normalization: (x - 5) / sqrt(4) = (x - 5) / 2
        expected_inp = (inp_raw - 5.0) / 2.0
        expected_tar = (tar_raw - 5.0) / 2.0

        assert torch.allclose(inp_norm, expected_inp), "Input normalization mismatch"
        assert torch.allclose(tar_norm, expected_tar), "Target normalization mismatch"
        print(f"  ✓ Normalization correct")

        print("  PASSED\n")
    finally:
        shutil.rmtree(tmpdir)


def test_dataloader():
    """Test DataLoader batching and shuffling."""
    print("=" * 60)
    print("TEST: DataLoader integration")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp()
    try:
        zarr_path, stats_path, N, T, C, H, W = create_test_db(tmpdir)

        ds = PrecomputedTrajectoryDataset(
            zarr_path=zarr_path, stats_path=stats_path,
            n_history=1, nfuture=0, normalize=True,
        )

        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

        batch_count = 0
        for inp, tar in loader:
            assert inp.shape[0] <= 4
            assert inp.shape[1:] == (C, H, W)
            assert tar.shape[1:] == (C, H, W)
            batch_count += 1

        expected_batches = (len(ds) + 3) // 4
        assert batch_count == expected_batches, f"Expected {expected_batches} batches, got {batch_count}"
        print(f"  ✓ DataLoader: {batch_count} batches, shapes correct")

        # Test no-op compatibility methods
        ds.set_num_examples(100)
        ds.set_initial_condition("random")
        print(f"  ✓ Compatibility methods (set_num_examples, set_initial_condition) work")

        print("  PASSED\n")
    finally:
        shutil.rmtree(tmpdir)


def test_cross_trajectory_boundary():
    """Verify that samples never cross trajectory boundaries."""
    print("=" * 60)
    print("TEST: No cross-trajectory contamination")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp()
    try:
        zarr_path, stats_path, N, T, C, H, W = create_test_db(tmpdir, N=3, T=5)

        ds = PrecomputedTrajectoryDataset(
            zarr_path=zarr_path, stats_path=stats_path,
            n_history=2, nfuture=1, normalize=False,
        )

        # Window = 2 + 1 + 1 = 4
        # Check every sample stays within its trajectory
        for idx in range(len(ds)):
            traj_i, t_start = ds.index[idx]
            inp, tar = ds[idx]

            # All values in inp should come from trajectory traj_i
            # inp: (2*C, H, W) = frames t_start, t_start+1
            # tar: (2, C, H, W) = frames t_start+2, t_start+3
            for c in range(C):
                # Frame 0 in history
                expected = traj_i * 1000 + t_start * 10 + c
                actual = inp[c, 0, 0].item()
                assert actual == expected, (
                    f"Sample {idx}: expected inp[{c}]={expected}, got {actual}"
                )
                # Frame 1 in history
                expected = traj_i * 1000 + (t_start + 1) * 10 + c
                actual = inp[C + c, 0, 0].item()
                assert actual == expected, (
                    f"Sample {idx}: expected inp[{C+c}]={expected}, got {actual}"
                )

        print(f"  ✓ All {len(ds)} samples verified: no cross-trajectory contamination")
        print("  PASSED\n")
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PrecomputedTrajectoryDataset Test Suite")
    print("=" * 60 + "\n")

    test_single_step()
    test_multi_history()
    test_multi_future()
    test_normalization()
    test_dataloader()
    test_cross_trajectory_boundary()

    print("=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)
