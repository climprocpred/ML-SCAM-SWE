# use pytorch3 conda environment
# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# Startup timing diagnostics
import time as _time
_startup_time = _time.time()
def _log_startup(msg):
    print(f"[STARTUP {_time.time() - _startup_time:.2f}s] {msg}", flush=True)

_log_startup("Beginning imports...")

import os, sys
import time
import argparse
from functools import partial

try:
    from tqdm import tqdm
    _log_startup("Imported tqdm")
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x
    _log_startup("tqdm not found, using dummy")

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler

_log_startup("Imported numpy/pandas")

import matplotlib.pyplot as plt
from wandb_plots import plot_rollout_snapshots, plot_rmse_vs_leadtime, plot_variance_vs_leadtime
_log_startup("Imported matplotlib and wandb_plots")

from torch_harmonics.examples import PdeDataset
from pde_dataset_extended import PdeDatasetExtended
from torch_harmonics.examples.losses import L1LossS2, SquaredL2LossS2, L2LossS2, W11LossS2
from laplacian_loss import LaplacianLossS2
from conservation_loss import ConservationLossS2
from CCC_loss import CCCLoss
from torch_harmonics import RealSHT
from torch_harmonics.plotting import plot_sphere
from spectral_loss import SpectralLossS2, GradientLossS2
_log_startup("Imported torch_harmonics")

# import baseline models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_registry import get_baseline_models
from normalization_utils import get_or_compute_stats

import math 

# Import Spectral PV Loss
try:
    from swe_davis.spectral_pv_loss import TwoStepPVResidual
    from swe_davis.spectral_mass_loss import TwoStepMassResidual
except ImportError:
    TwoStepPVResidual = None
    TwoStepMassResidual = None
    print("Warning: Could not import TwoStepPVResidual from swe_davis.spectral_pv_loss")
_log_startup("Imported model_registry")

# wandb logging
try:
    import wandb
    _log_startup("Imported wandb")
except:
    wandb = None
    _log_startup("wandb not available")

# Get PBS job ID for logging
PBS_JOBID = os.environ.get("PBS_JOBID", "N/A")
print(f"PBS Job ID: {PBS_JOBID}", flush=True)


# helper routine for counting number of paramerters in model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# convenience function for logging weights and gradients
def log_weights_and_grads(model, iters=1):
    root_path = os.path.join(os.path.dirname(__file__), "weights_and_grads")

    weights_and_grads_fname = os.path.join(root_path, f"weights_and_grads_step{iters:03d}.tar")
    print(weights_and_grads_fname)

    weights_dict = {k: v for k, v in model.named_parameters()}
    grad_dict = {k: v.grad for k, v in model.named_parameters()}

    store_dict = {"iteration": iters, "grads": grad_dict, "weights": weights_dict}
    torch.save(store_dict, weights_and_grads_fname)


# rolls out the FNO and compares to the classical solver
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
    # Use a finite radius for the Cosine Bell to ensure exact zero support
    radius_bump = 40.0 * math.pi / 180.0 
    amp = solver.hamp * 2.0 # Reduced amplitude 
    
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

def autoregressive_inference(
    model,
    dataset,
    loss_fn,
    metrics_fns,
    path_root,
    nsteps,
    autoreg_steps=10,
    nskip=1,
    plot_channel=0,
    nics=10,
    device=torch.device("cpu"),
    run_tag=None,
    n_history=1,  # Number of past timesteps model expects (1=single-step, 3=AB3-like)
    precomputed_dataset=None,
):

    model.eval()

    if nics <= 0:
        print("Skipping autoregressive inference (nics=0)")
        return torch.tensor([]), {}, [], []

    # make output
    if not os.path.isdir(path_root):
        os.makedirs(path_root, exist_ok=True)

    # accumulation buffers for losses, metrics and runtimes
    losses = torch.zeros(nics, dtype=torch.float32, device=device)
    metrics = {}
    for metric in metrics_fns:
        metrics[metric] = torch.zeros(nics, dtype=torch.float32, device=device)
    model_times = torch.zeros(nics, dtype=torch.float32, device=device)
    solver_times = torch.zeros(nics, dtype=torch.float32, device=device)

    # accumulation buffers for the power spectrum
    prd_mean_coeffs = []
    ref_mean_coeffs = []

    # Note: We need to accumulate fields for NetCDF saving
    # shape: (nics, steps, C, lat, lon)
    all_preds = []
    all_truth = []
    
    # New buffers for u and v components
    all_preds_u = []
    all_preds_v = []
    all_truth_u = []
    all_truth_v = []
    
    for iic in range(nics):
        # Use Galewsky IC for first sample, Gaussian Bump for second, random for rest
        use_precomputed_truth = False
        states_chunk = None

        if iic == 0:
            ic = dataset.solver.galewsky_initial_condition()
        elif iic == 1:
            ic = gaussian_bump_initial_condition(dataset.solver)
        else:
            if precomputed_dataset is not None:
                # Use precomputed trajectory
                use_precomputed_truth = True
                
                # Sample random trajectory and start time
                # Ensure we have enough future steps: t_start + autoreg_steps <= T_total - 1
                # So max t_start = T_total - 1 - autoreg_steps
                max_start = precomputed_dataset.T_total - 1 - autoreg_steps
                
                # If window is too large for trajectory, this might fail, but T_total should be large
                if max_start < 0:
                     raise ValueError(f"Trajectory length {precomputed_dataset.T_total} too short for validation rollout {autoreg_steps}")
                
                traj_idx = torch.randint(0, precomputed_dataset.N, (1,)).item()
                t_idx = torch.randint(0, max_start + 1, (1,)).item()
                
                # Load chunk: [t_idx, t_idx + autoreg_steps + 1) -> length autoreg_steps + 1
                # states are (N, T, C, H, W)
                # We need to handle potential retries here too if Zarr is flaky, but dataset class has it.
                # However we are accessing .states directly here.
                # Let's trust Zarr combined with the earlier retry patch or just do a simple try/loop if needed.
                # Since we are in validation, a crash is annoying but less critical than training loop. 
                # We'll access via the dataset object if possible or just direct slice. 
                # Direct slice is easiest given we need a custom window length.
                
                # shape: (autoreg_steps + 1, C, H, W)
                states_chunk = torch.from_numpy(precomputed_dataset.states[traj_idx, t_idx : t_idx + autoreg_steps + 1])
                states_chunk = states_chunk.float().to(device)
                
                # Initial state (step 0)
                state0_grid = states_chunk[0] # (C, H, W)
                
                # Convert to spectral IC for model input compatibility (if model needs spectral)
                # But here we just need Grid state for normalization
                # We do need 'ic' (spectral) if we were running solver, but we aren't.
                # However, consistent with below, we might want 'ic' for some reason? 
                # Actually, 'uspec' is derived from 'ic'. 'uspec' is used for solver stepping.
                # We won't step solver, so we don't strictly need 'ic' or 'uspec' for truth evolution.
                # But we do need 'ic' if we want to calculate initial U/V using solver methods (grid2spec)
                ic = dataset.solver.grid2spec(state0_grid)
                
            else:
                ic = dataset.solver.random_initial_condition(mach=0.2)

        inp_mean = dataset.inp_mean
        inp_var = dataset.inp_var

        # Initial Condition (step 0) - single normalized state (C, H, W)
        if use_precomputed_truth:
             # state0_grid already extracted
             pass
        else:
             state0_grid = dataset.solver.spec2grid(ic)
             
        state0 = (state0_grid - inp_mean) / torch.sqrt(inp_var)
        state0 = state0.unsqueeze(0)  # (1, C, H, W)
        
        if not use_precomputed_truth:
            uspec = ic.clone()

        # Storage for this IC
        ic_preds = [state0[0].detach().cpu()]
        ic_truth = [state0[0].detach().cpu()]
        
        # Calculate initial u/v for prediction (from ground truth IC)
        # dataset.solver.getuv expects spectral vorticity and divergence
        # ic is full spectral state (height, vorticity, divergence)
        uv_ic = dataset.solver.getuv(ic[1:]) # (2, lat, lon)
        ic_preds_u = [uv_ic[0].detach().cpu()]
        ic_preds_v = [uv_ic[1].detach().cpu()]
        ic_truth_u = [uv_ic[0].detach().cpu()]
        ic_truth_v = [uv_ic[1].detach().cpu()]

        # add IC to power spectrum series
        prd_coeffs = [dataset.sht(state0[0, plot_channel]).detach().cpu().clone()]
        ref_coeffs = [prd_coeffs[0].clone()]

        # Initialize history buffer for multi-step models
        # For n_history=3: buffer = [state0, state0, state0] initially (like AB3's Forward Euler)
        # Each state is (1, C, H, W), and they get stacked to (1, n_history*C, H, W)
        if n_history > 1:
            history_buffer = [state0.clone() for _ in range(n_history)]
        else:
            # For 1-step, we rely on 'prd' variable
            pass # history_buffer = None

        # ML model autoregressive rollout
        start_time = time.time()
        # Initialize prd for single-step logic
        prd = state0 
        
        for i in range(1, autoreg_steps + 1):
            # Build model input based on n_history
            if n_history > 1:
                # Stack history: [oldest, ..., newest] -> (1, n_history*C, H, W)
                model_input = torch.cat(history_buffer, dim=1)
            else:
                # Single-step: input is just the previous prediction
                model_input = prd
            
            # Evaluate the ML model
            prd = model(model_input)  # Output: (1, C, H, W)
            
            # Update history buffer (sliding window)
            if n_history > 1:
                history_buffer = history_buffer[1:] + [prd.clone()]
            
            # Store prediction (only the output C channels, not the stacked input)
            ic_preds.append(prd[0].detach().cpu())
            prd_coeffs.append(dataset.sht(prd[0, plot_channel]).detach().cpu().clone())
            
            # Calculate and store u/v for prediction
            # 1. Denormalize to get physical grid values
            prd_phys = prd[0] * torch.sqrt(inp_var) + inp_mean
            # 2. Convert to spectral to get vorticity/divergence coeffs
            prd_spec = dataset.solver.grid2spec(prd_phys)
            # 3. Calculate u/v from vorticity/divergence (channels 1 and 2)
            uv_prd = dataset.solver.getuv(prd_spec[1:])
            ic_preds_u.append(uv_prd[0].detach().cpu())
            ic_preds_v.append(uv_prd[1].detach().cpu())

        model_times[iic] = time.time() - start_time

        # classical model (or precomputed truth)
        start_time = time.time()
        
        if use_precomputed_truth:
            # Use precomputed trajectory slice
            # states_chunk has shape (steps+1, C, H, W)
            # We already used index 0 for IC. Now iterate 1..steps
            for i in range(1, autoreg_steps + 1):
                # Get grid state directly
                ref_phys = states_chunk[i] # (C, H, W)
                
                # Normalize for metrics/loss
                ref = (ref_phys - inp_mean) / torch.sqrt(inp_var)
                ic_truth.append(ref.detach().cpu())
                
                ref_coeffs.append(dataset.sht(ref[plot_channel]).detach().cpu().clone())
                
                # Calculate U/V from physical grid state
                # Need consistent usage of solver utils
                ref_spec = dataset.solver.grid2spec(ref_phys)
                uv_ref = dataset.solver.getuv(ref_spec[1:])
                ic_truth_u.append(uv_ref[0].detach().cpu())
                ic_truth_v.append(uv_ref[1].detach().cpu())
                
        else:
            # Run solver on-the-fly
            for i in range(1, autoreg_steps + 1):
                # advance classical model
                uspec = dataset.solver.timestep(uspec, nsteps)
                ref = (dataset.solver.spec2grid(uspec) - inp_mean) / torch.sqrt(inp_var)
                
                # Store truth
                ic_truth.append(ref.detach().cpu())

                ref_coeffs.append(dataset.sht(ref[plot_channel]).detach().cpu().clone())
                
                # Calculate and store u/v for truth
                # uspec already has physical spectral coefficients
                uv_ref = dataset.solver.getuv(uspec[1:])
                ic_truth_u.append(uv_ref[0].detach().cpu())
                ic_truth_v.append(uv_ref[1].detach().cpu())

        solver_times[iic] = time.time() - start_time
        
        # Append to main storage
        all_preds.append(torch.stack(ic_preds)) # (steps, C, H, W)
        all_truth.append(torch.stack(ic_truth))
        all_preds_u.append(torch.stack(ic_preds_u))
        all_preds_v.append(torch.stack(ic_preds_v))
        all_truth_u.append(torch.stack(ic_truth_u))
        all_truth_v.append(torch.stack(ic_truth_v))

        # compute power spectrum and add it to the buffers
        prd_mean_coeffs.append(torch.stack(prd_coeffs, 0))
        ref_mean_coeffs.append(torch.stack(ref_coeffs, 0))

        # Compute metrics on the LAST step (as was done originally)
        # ref re-calculated/extracted for consistency with original loss logic
        ref_last = ic_truth[-1].to(device).unsqueeze(0) 
        prd_last = ic_preds[-1].to(device).unsqueeze(0)
        
        losses[iic] = loss_fn(prd_last, ref_last)
        for metric in metrics_fns:
            metric_buff = metrics[metric]
            metric_fn = metrics_fns[metric]
            metric_buff[iic] = metric_fn(prd_last, ref_last)

    # compute the averaged powerspectra 
    with torch.no_grad():
        prd_mean_coeffs = torch.stack(prd_mean_coeffs, dim=0).abs().pow(2).mean(dim=0)
        ref_mean_coeffs = torch.stack(ref_mean_coeffs, dim=0).abs().pow(2).mean(dim=0)

        prd_mean_coeffs[..., 1:] *= 2.0
        ref_mean_coeffs[..., 1:] *= 2.0
        prd_mean_ps = prd_mean_coeffs.sum(dim=-1).contiguous()
        ref_mean_ps = ref_mean_coeffs.sum(dim=-1).contiguous()

    # SAVE NETCDF
    import xarray as xr
    
    # Stack all data: (nics, steps, C, H, W)
    preds_tensor = torch.stack(all_preds).numpy()
    truth_tensor = torch.stack(all_truth).numpy()
    
    # Stack UV data: (nics, steps, H, W) -> (nics, steps, 1, H, W)
    preds_u = torch.stack(all_preds_u).unsqueeze(2).numpy()
    preds_v = torch.stack(all_preds_v).unsqueeze(2).numpy()
    truth_u = torch.stack(all_truth_u).unsqueeze(2).numpy()
    truth_v = torch.stack(all_truth_v).unsqueeze(2).numpy()
    
    # Concatenate along channel dimension (axis 2)
    # Resulting shape: (nics, steps, 5, H, W)
    preds_combined = np.concatenate([preds_tensor, preds_u, preds_v], axis=2)
    truth_combined = np.concatenate([truth_tensor, truth_u, truth_v], axis=2)
    
    # Define coords
    times = np.arange(autoreg_steps + 1)
    samples = np.arange(nics)
    # chans = np.arange(dataset.nlat) if preds_tensor.shape[2] == dataset.nlat else np.arange(preds_tensor.shape[2]) # Usually 3 chans
    chans = ["geopotential_height", "vorticity", "divergence", "u", "v"]
    lats = np.linspace(-90, 90, dataset.nlat) # Approx, ideally use dataset.grid
    lons = np.linspace(0, 360, dataset.nlon, endpoint=False)
    
    ds = xr.Dataset(
        data_vars={
            "prediction": (("sample", "time", "channel", "lat", "lon"), preds_combined),
            "truth": (("sample", "time", "channel", "lat", "lon"), truth_combined),
        },
        coords={
            "sample": samples,
            "time": times,
            "channel": chans,
            "lat": lats,
            "lon": lons,
        }
    )
    
    scratch_dir = f"/glade/derecho/scratch/{os.environ['USER']}/TH_SWE_output"
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir, exist_ok=True)
    
    # Extract model name from path_root (.../model_name/figures)
    model_name = os.path.basename(os.path.dirname(path_root))
    # Build filename with optional run_tag
    if run_tag:
        save_path = os.path.join(scratch_dir, f"{model_name}_{run_tag}_validation_steps{autoreg_steps}.nc")
    else:
        save_path = os.path.join(scratch_dir, f"{model_name}_validation_steps{autoreg_steps}.nc")
    print(f"Saving NetCDF to {save_path}...")
    ds.to_netcdf(save_path)
    print(f"Save complete. Output file is located at: {save_path}")

    return losses, metrics, model_times, solver_times


# training function
def train_model(
    model,
    dataloader,
    loss_fn,
    metrics_fns,
    optimizer,
    gscaler,
    scheduler=None,
    nepochs=20,
    nfuture=0,
    num_examples=256,
    num_valid=0,
    amp_mode="none",
    log_grads=0,
    logging=True,
    device=torch.device("cpu"),
    n_history=1,  # Number of past timesteps model expects
    noise_injection_sigma=0.0,
    noise_injection_rate=0.0,
    plot_callback=None,
    scratch_dir=None,
    time_diff2_lambda=0.0,
):
   
    train_start = time.time()

    # set AMP type
    amp_dtype = torch.float32
    if amp_mode == "fp16":
        amp_dtype = torch.float16
    elif amp_mode == "bf16":
        amp_dtype = torch.bfloat16

    # count iterations
    iters = 0
    valid_loss = 0.0 # Initialize valid_loss to prevent UnboundLocalError if nepochs=0
    best_valid_loss = float('inf')

    for epoch in range(nepochs):

        # time each epoch
        epoch_start = time.time()

        if hasattr(dataloader.dataset, 'set_initial_condition'):
            dataloader.dataset.set_initial_condition("random")
        if hasattr(dataloader.dataset, 'set_num_examples'):
            dataloader.dataset.set_num_examples(num_examples)

        # get the solver for its convenience functions
        solver = getattr(dataloader.dataset, 'solver', None)

        # do the training
        accumulated_loss = 0
        accumulated_l2_loss = 0
        accumulated_lap_loss = 0
        accumulated_cons_loss = 0
        accumulated_ccc_loss = 0
        accumulated_spectral_loss = 0
        accumulated_gradient_loss = 0
        accumulated_spectral_pv_loss = 0
        accumulated_spectral_pv_loss = 0
        accumulated_mass_loss = 0
        accumulated_time_diff2_loss = 0
        model.train()

        for batch_idx, (inp, tar) in enumerate(dataloader):

            inp, tar = inp.to(device), tar.to(device)

            # --- Noise Injection Logic ---
            if noise_injection_sigma > 0 and noise_injection_rate > 0 and inp.size(0) >= 2:
                batch_size = inp.size(0)
                half_batch = batch_size // 2
                
                # Iterate over the first half of the batch
                for i in range(half_batch):
                    # With probability `noise_injection_rate`, duplicate sample `i` to `i + half_batch`
                    if torch.rand(1).item() < noise_injection_rate:
                        target_idx = i + half_batch
                        
                        # Duplicate input
                        inp[target_idx] = inp[i].clone()
                        
                        # Duplicate target (Ground truth for noisy input is clean target)
                        if tar.dim() == 5: # Trajectory (B, T, C, H, W)
                             tar[target_idx] = tar[i].clone()
                        else: # Single step (B, C, H, W)
                             tar[target_idx] = tar[i].clone()
                        
                        # Add noise to the duplicate input
                        noise = torch.randn_like(inp[target_idx]) * noise_injection_sigma
                        inp[target_idx] += noise
            # -----------------------------

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_mode != "none")):

                # Check if tar is a trajectory (5D: B, T, C, H, W) or single target (4D: B, C, H, W)
                if tar.dim() == 5:
                    # Multi-step trajectory: accumulate loss at each step
                    loss = 0
                    num_steps = tar.shape[1]
                    prev_prev_state = None
                    
                    if batch_idx == 0 and epoch == 0:
                        print(f"DEBUG: Autoregressive training running. tar shape: {tar.shape}, num_steps: {num_steps}, dataset: {dataloader.dataset.__class__.__name__}")
                    
                    if hasattr(loss_fn, 'reset_stats'):
                        loss_fn.reset_stats()
                    
                    # For multi-history: need to maintain history buffer
                    if n_history > 1:
                        # inp shape: (B, n_history*C, H, W), extract per-timestep chunks
                        B, total_C, H, W = inp.shape
                        physical_chans = total_C // n_history  # 3 for SWE
                        # Initialize history buffer with input timesteps
                        history_buffer = [inp[:, i*physical_chans:(i+1)*physical_chans] for i in range(n_history)]
                        
                        for step in range(num_steps):
                            model_input = torch.cat(history_buffer, dim=1)
                            prd = model(model_input)  # Output: (B, C, H, W)
                            step_target = tar[:, step]
                            
                            # Identify prev_target, prev_prev_target for loss
                            prev_target = None
                            prev_prev_target = None
                            
                            if step == 0:
                                # For step 0, prev_target is last frame of input
                                prev_target = history_buffer[-1]
                                if n_history > 1:
                                     # prev_prev is second to last
                                     prev_prev_target = history_buffer[-2]
                            else:
                                prev_target = tar[:, step-1]
                                if step == 1:
                                    prev_prev_target = history_buffer[-1]
                                else:
                                    prev_prev_target = tar[:, step-2]
                            
                            # For step 0, prev_state is the last input frame (from history or inp)
                            # If step > 0, prev_state is the previous prediction
                            if step == 0:
                                # history_buffer[-1] is the most recent frame
                                prev_state = history_buffer[-1]
                                if n_history > 1:
                                    prev_prev_state = history_buffer[-2]
                            else:
                                prev_state = history_buffer[-1] 
                                if n_history > 1:
                                    prev_prev_state = history_buffer[-2]
                                else:
                                    # For n_history=1, we need to manually track prev_prev if available
                                    # But in this branch (n_history>1) we have history_buffer.
                                    pass
                            
                            # Let's simplify: pass model_input (which contains history)
                            # CombinedLoss can extract the last frame.
                            loss = loss + loss_fn(prd, step_target, 
                                                prev_state=prev_state, 
                                                prev_prev_state=prev_prev_state,
                                                prev_target=prev_target,
                                                prev_prev_target=prev_prev_target,
                                                step=step)
                            
                            # Update prev_prev_state for NEXT step (it becomes prev_state of next step)
                            if n_history > 1:
                                prev_prev_state = history_buffer[-1]
                            else:
                                prev_prev_state = model_input

                            # Update history buffer (sliding window)
                            history_buffer = history_buffer[1:] + [prd]
                    else:
                        # Single-history: original behavior
                        prd = inp
                        prev_target = None
                        prev_prev_target = None
                        
                        if hasattr(loss_fn, 'reset_stats'):
                            loss_fn.reset_stats()
                            
                        for step in range(num_steps):
                            model_input = prd # Save previous state
                            
                            # Update targets
                            step_target = tar[:, step]
                            if step == 0:
                                prev_target = inp
                                prev_prev_target = None # Can't know t-2 easily without history
                            else:
                                prev_target = tar[:, step-1]
                                if step == 1:
                                    prev_prev_target = inp
                                else:
                                    prev_prev_target = tar[:, step-2]

                            prd = model(prd)
                            
                            if step == 0:
                                prev_prev_state = None
                            else:
                                prev_prev_state = model_input
                            
                            loss = loss + loss_fn(prd, step_target, 
                                                prev_state=model_input, 
                                                prev_prev_state=prev_prev_state,
                                                prev_target=prev_target,
                                                prev_prev_target=prev_prev_target,
                                                step=step)
                    
                    # Average over steps for consistent loss magnitude
                    loss = loss / num_steps
                else:
                    # Single-step target: original behavior with optional nfuture rollout
                    if n_history > 1 and nfuture > 0:
                        # Multi-history with rollout: need history buffer
                        B, total_C, H, W = inp.shape
                        physical_chans = total_C // n_history
                        history_buffer = [inp[:, i*physical_chans:(i+1)*physical_chans] for i in range(n_history)]
                        
                        if hasattr(loss_fn, 'reset_stats'):
                            loss_fn.reset_stats()
                            
                        for _ in range(nfuture + 1):  # +1 for initial prediction
                            model_input = torch.cat(history_buffer, dim=1)
                            prd = model(model_input)
                            history_buffer = history_buffer[1:] + [prd]
                        # So we pass the last input as prev_state logic?
                        # If nfuture > 0, we did multiple steps. The last prd is result of last model_input.
                        # For single-step loss with rollout (nfuture>0), we don't have intermediate targets in this branch
                        # But if we want time_diff2 loss on the FINAL step, we need u_{T-1} and u_{T-2}
                        # The loop above updates history_buffer.
                        # history_buffer[-2] would be u_{T-2} effectively?
                        # This branch is rarely used with time_diff2 but let's try to support it.
                        prev_prev = None
                        if nfuture >= 1 and n_history > 1:
                             prev_prev = history_buffer[-2] 

                        loss = loss_fn(prd, tar, prev_state=model_input, prev_prev_state=prev_prev, step=nfuture)
                    else:
                        # Standard single-step or single-history
                        model_input = inp
                        prd = model(inp)
                        prev_prev_state = None
                        prev_state = model_input
                        
                        if hasattr(loss_fn, 'reset_stats'):
                            loss_fn.reset_stats()
                        
                        # Store initial state as prev_prev if nfuture >= 1
                        # Store initial state as prev_prev if nfuture >= 1
                        if nfuture >= 1:
                             prev_prev_state = prev_state # u_t
                             prev_state = prd # u_{t+1} (after first step)
                             
                             # Track ground truth history
                             # For step 0 (implicit above): prev_target=inp, target=tar[:,0]
                             # For step 1 (first iter below): prev_target=tar[:,0], etc.
                             pass

                        for step in range(nfuture):
                            # logic: 
                            # step 0 (already done above): inp -> prd(t+1). prev=inp. prev_prev=None.
                            # step 1 (loop): model_input=prd(t+1). prd(t+2). prev=prd(t+1). prev_prev=inp.
                            
                            # We need to correctly identify prev_target and prev_prev_target from Ground Truth
                            # The 'tar' variable in this branch is (B, C, H, W) - single step target.
                            # Wait, 'tar' is single step. 'nfuture' means we rollout WITHOUT gradient supervision on intermediate steps?
                            # No, usually nfuture training involves multi-step loss?
                            # In this specific branch (else: single step target), 'tar' is just the immediate next step.
                            # So we CANNOT calculate diff2_gt effectively because we don't have t+2 ground truth.
                            # So this branch only supports diff2 loss if nfuture=0 (default).
                            pass
                            
                            if step > 0:
                                 prev_prev_state = prev_state
                                 prev_state = prd

                            model_input = prd
                            prd = model(prd)
                        
                        loss = loss_fn(prd, tar, prev_state=model_input, prev_prev_state=prev_prev_state, prev_target=None, prev_prev_target=None, step=nfuture)

            accumulated_loss += loss.item() * inp.size(0)
            
            # For trajectory generation, we accumulated raw components across rollout steps. Scale them back to step-averages.
            div = tar.shape[1] if tar.dim() == 5 else 1.0
            
            # Accumulate loss components if available
            if hasattr(loss_fn, 'last_l2_loss'):
                accumulated_l2_loss += (loss_fn.last_l2_loss / div) * inp.size(0)
            if hasattr(loss_fn, 'last_lap_loss'):
                accumulated_lap_loss += (loss_fn.last_lap_loss / div) * inp.size(0)
            if hasattr(loss_fn, 'last_cons_loss'):
                accumulated_cons_loss += (loss_fn.last_cons_loss / div) * inp.size(0)
            if hasattr(loss_fn, 'last_ccc_loss'):
                accumulated_ccc_loss += (loss_fn.last_ccc_loss / div) * inp.size(0)
            if hasattr(loss_fn, 'last_spectral_loss'):
                accumulated_spectral_loss += (loss_fn.last_spectral_loss / div) * inp.size(0)
            if hasattr(loss_fn, 'last_gradient_loss'):
                accumulated_gradient_loss += (loss_fn.last_gradient_loss / div) * inp.size(0)
            if hasattr(loss_fn, 'last_spectral_pv_loss'):
                accumulated_spectral_pv_loss += (loss_fn.last_spectral_pv_loss / div) * inp.size(0)
            if hasattr(loss_fn, 'last_mass_loss'):
                accumulated_mass_loss += (loss_fn.last_mass_loss / div) * inp.size(0)
            if hasattr(loss_fn, 'last_time_diff2_loss'):
                accumulated_time_diff2_loss += (loss_fn.last_time_diff2_loss / div) * inp.size(0)

            optimizer.zero_grad(set_to_none=True)
            gscaler.scale(loss).backward()

            if log_grads and iters % log_grads == 0:
                log_weights_and_grads(model, iters=iters)

            gscaler.step(optimizer)
            gscaler.update()

            iters += 1

        accumulated_loss = accumulated_loss / len(dataloader.dataset)
        # Average loss components and store for logging
        if hasattr(loss_fn, 'last_l2_loss'):
            loss_fn.last_l2_loss = accumulated_l2_loss / len(dataloader.dataset)
        if hasattr(loss_fn, 'last_lap_loss'):
            loss_fn.last_lap_loss = accumulated_lap_loss / len(dataloader.dataset)
        if hasattr(loss_fn, 'last_cons_loss'):
            loss_fn.last_cons_loss = accumulated_cons_loss / len(dataloader.dataset)
        if hasattr(loss_fn, 'last_ccc_loss'):
            loss_fn.last_ccc_loss = accumulated_ccc_loss / len(dataloader.dataset)
        if hasattr(loss_fn, 'last_spectral_loss'):
            loss_fn.last_spectral_loss = accumulated_spectral_loss / len(dataloader.dataset)
        if hasattr(loss_fn, 'last_gradient_loss'):
            loss_fn.last_gradient_loss = accumulated_gradient_loss / len(dataloader.dataset)
        if hasattr(loss_fn, 'last_spectral_pv_loss'):
            loss_fn.last_spectral_pv_loss = accumulated_spectral_pv_loss / len(dataloader.dataset)
        if hasattr(loss_fn, 'last_mass_loss'):
            loss_fn.last_mass_loss = accumulated_mass_loss / len(dataloader.dataset)
        if hasattr(loss_fn, 'last_time_diff2_loss'):
            loss_fn.last_time_diff2_loss = accumulated_time_diff2_loss / len(dataloader.dataset)

        # Skip validation if num_valid is 0, use training loss instead
        if num_valid > 0:
            dataloader.dataset.set_initial_condition("random")
            dataloader.dataset.set_num_examples(num_valid)

            # eval mode
            model.eval()

            # prepare loss buffer for validation loss
            valid_loss = torch.zeros(2, dtype=torch.float32, device=device)

            # prepare metrics buffer for accumulation of validation metrics
            valid_metrics = {}
            for metric in metrics_fns:
                valid_metrics[metric] = torch.zeros(2, dtype=torch.float32, device=device)

            # perform validation
            with torch.no_grad():
                for inp, tar in dataloader:
                    # Check if tar is a trajectory (5D: B, T, C, H, W) or single target (4D: B, C, H, W)
                    if tar.dim() == 5:
                        # Multi-step trajectory: accumulate loss at each step
                        loss = 0
                        prd = inp
                        num_steps = tar.shape[1]
                        for step in range(num_steps):
                            prd = model(prd)
                            step_target = tar[:, step]  # (B, C, H, W)
                            loss = loss + loss_fn(prd, step_target, step=step)
                        # Average over steps
                        loss = (loss / num_steps).item()
                        # For metrics, use final prediction vs final target
                        final_target = tar[:, -1]
                    else:
                        # Single-step target: original behavior
                        prd = model(inp)
                        for _ in range(nfuture):
                            prd = model(prd)
                        loss = loss_fn(prd, tar, step=nfuture).item()
                        final_target = tar

                    valid_loss[0] += loss * inp.size(0)
                    valid_loss[1] += inp.size(0)

                    for metric in metrics_fns:
                        metric_buff = valid_metrics[metric]
                        metric_fn = metrics_fns[metric]
                        metric_buff[0] += metric_fn(prd, final_target) * inp.size(0)
                        metric_buff[1] += inp.size(0)

            valid_loss = (valid_loss[0] / valid_loss[1]).item()
            for metric in valid_metrics:
                valid_metrics[metric] = (valid_metrics[metric][0] / valid_metrics[metric][1]).item()
        else:
            # No validation - use training loss for scheduler and checkpointing
            valid_loss = accumulated_loss
            valid_metrics = {}

        if scheduler is not None:
            scheduler.step(valid_loss)

        epoch_time = time.time() - epoch_start

        if logging:
            print(f"--------------------------------------------------------------------------------")
            print(f"Epoch {epoch} summary:")
            print(f"time taken: {epoch_time:.2f}")
            print(f"accumulated training loss: {accumulated_loss}")
            
            # Detailed Loss Breakdown Table
            if hasattr(loss_fn, 'last_channel_losses'):
                print("\n  Loss Breakdown:")
                print(f"  {'Component':<20} | {'Raw Value':<12} | {'Lambda':<10} | {'Weighted':<12} | {'% Total':<8}")
                print("  " + "-"*75)
                
                loss_components = [
                    ("L2", "last_l2_loss", "l2_lambda"),
                    ("Laplacian", "last_lap_loss", "lap_lambda"),
                    ("Conservation", "last_cons_loss", "cons_lambda"),
                    ("CCC", "last_ccc_loss", "ccc_lambda"),
                    ("Spectral", "last_spectral_loss", "spectral_lambda"),
                    ("Gradient", "last_gradient_loss", "gradient_lambda"),
                    ("UV", "last_uv_loss", "uv_lambda"),
                    ("Spectral PV", "last_spectral_pv_loss", "spectral_pv_lambda"),
                    ("Spectral Mass", "last_mass_loss", "mass_lambda"),
                    ("Time Diff2", "last_time_diff2_loss", "time_diff2_lambda"),
                ]
                
                total_check = 0.0
                component_data = []

                # First pass: collect data and total
                for name, attr_loss, attr_lambda in loss_components:
                    if hasattr(loss_fn, attr_loss) and hasattr(loss_fn, attr_lambda):
                        raw_val = getattr(loss_fn, attr_loss)
                        lam_val = getattr(loss_fn, attr_lambda)
                        if lam_val > 0:
                            weighted_val = raw_val * lam_val
                            component_data.append((name, raw_val, lam_val, weighted_val))
                            total_check += weighted_val
                
                # Second pass: print rows
                for name, raw_val, lam_val, weighted_val in component_data:
                    pct = (weighted_val / total_check * 100) if total_check > 0 else 0
                    print(f"  {name:<20} | {raw_val:<12.6f} | {lam_val:<10.2e} | {weighted_val:<12.6f} | {pct:<6.1f}%")
                print("  " + "-"*75 + "\n")

            # Print per-variable losses
            if hasattr(loss_fn, 'last_channel_losses'):
                ch_losses = loss_fn.last_channel_losses
                print(f"  per-variable L2: height={ch_losses['height']:.6f}, vorticity={ch_losses['vorticity']:.6f}, divergence={ch_losses['divergence']:.6f}")
            if num_valid > 0:
                print(f"validation loss: {valid_loss}")
                for metric in valid_metrics:
                    print(f"{metric}: {valid_metrics[metric]}")

        # Check and save best checkpoint AFTER printing epoch summary
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if hasattr(model, 'best_checkpoint_path'):
                if logging:
                    print(f"Training loss improved to {best_valid_loss:.6f}. Saving best checkpoint...")
                torch.save(model.state_dict(), model.best_checkpoint_path)
        
        # Always save recent checkpoint (every epoch)
        if hasattr(model, 'recent_checkpoint_path'):
            torch.save(model.state_dict(), model.recent_checkpoint_path)

        # Save a copy to scratch directory if provided
        if scratch_dir is not None:
            if not os.path.exists(scratch_dir):
                os.makedirs(scratch_dir, exist_ok=True)
            scratch_checkpoint_path = os.path.join(scratch_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(model.state_dict(), scratch_checkpoint_path)


        # Periodic plot logging (commit=False, so plots merge into the metrics log below)
        if plot_callback is not None:
            plot_callback(model, epoch_label=f"pretrain_ep{epoch}")

        if logging:
            if wandb.run is not None:
                current_lr = optimizer.param_groups[0]["lr"]
                log_dict = {"loss": accumulated_loss, "learning rate": current_lr}
                # Log separate loss components if available
                if hasattr(loss_fn, 'last_l2_loss'):
                    log_dict["L2 loss"] = loss_fn.last_l2_loss
                if hasattr(loss_fn, 'last_lap_loss'):
                    log_dict["Laplacian loss"] = loss_fn.last_lap_loss
                if hasattr(loss_fn, 'last_cons_loss'):
                    log_dict["Conservation loss"] = loss_fn.last_cons_loss
                if hasattr(loss_fn, 'last_ccc_loss'):
                    log_dict["CCC loss"] = loss_fn.last_ccc_loss
                if hasattr(loss_fn, 'last_spectral_loss'):
                    log_dict["Spectral loss"] = loss_fn.last_spectral_loss
                if hasattr(loss_fn, 'last_gradient_loss'):
                    log_dict["Gradient loss"] = loss_fn.last_gradient_loss
                if hasattr(loss_fn, 'last_uv_loss'):
                    log_dict["UV loss"] = loss_fn.last_uv_loss
                if hasattr(loss_fn, 'last_spectral_pv_loss'):
                    log_dict["Spectral PV loss"] = loss_fn.last_spectral_pv_loss
                if hasattr(loss_fn, 'last_mass_loss'):
                    log_dict["Spectral Mass loss"] = loss_fn.last_mass_loss
                if hasattr(loss_fn, 'last_time_diff2_loss'):
                    log_dict["Time Diff2 loss"] = loss_fn.last_time_diff2_loss
                if num_valid > 0:
                    log_dict["validation loss"] = valid_loss
                    for metric in valid_metrics:
                        log_dict[metric] = valid_metrics[metric]
                wandb.log(log_dict)

    train_time = time.time() - train_start

    print(f"--------------------------------------------------------------------------------")
    print(f"done. Training took {train_time}.")
    return valid_loss

# Curriculum rollout training: train on final step after variable-length gradient-free rollouts
def train_curriculum_rollout(
    model,
    dataset,
    loss_fn,
    optimizer,
    gscaler,
    scheduler=None,
    nepochs=50,
    num_examples=256,
    batch_size=4,
    amp_mode="none",
    max_steps_schedule=None,  # List: max_steps for each epoch, e.g., [1, 2, 4, 8, 16]
    logging=True,
    device=torch.device("cpu"),
    n_history=1,  # Number of past timesteps model expects (1=single-step, 3=AB3-like)
    noise_injection_sigma=0.0,
    noise_injection_rate=0.0,
    plot_callback=None,
    scratch_dir=None,
):
    """
    Curriculum rollout training mode with multi-history support.
    
    For each batch:
    1. Generate random IC -> normalize
    2. Run ML model for n_steps WITHOUT gradients (detached rollout)
       - For n_history>1: maintain sliding window history buffer
    3. Convert final ML state to spectral -> run solver 1 step -> normalize (ground truth)
    4. Run ML model 1 step WITH gradients (prediction)
    5. Compute loss -> backprop
    
    n_steps is randomly sampled from [0, current_max_steps] each batch.
    current_max_steps follows the schedule, increasing each epoch.
    """
    train_start = time.time()
    
    # Default schedule: linear ramp from 1 to 32 over nepochs
    if max_steps_schedule is None:
        max_steps_schedule = [min(1 + epoch, 32) for epoch in range(nepochs)]
    
    # Extend schedule if shorter than nepochs
    while len(max_steps_schedule) < nepochs:
        max_steps_schedule.append(max_steps_schedule[-1])
    
    # Set AMP type
    amp_dtype = torch.float32
    if amp_mode == "fp16":
        amp_dtype = torch.float16
    elif amp_mode == "bf16":
        amp_dtype = torch.bfloat16
    
    solver = dataset.solver
    inp_mean = dataset.inp_mean
    inp_var = dataset.inp_var
    nsteps = dataset.nsteps  # Solver steps per ML step
    
    best_loss = float('inf')
    
    for epoch in range(nepochs):
        epoch_start = time.time()
        current_max_steps = max_steps_schedule[epoch]
        
        model.train()
        accumulated_loss = 0.0
        accumulated_l2_loss = 0.0
        accumulated_lap_loss = 0.0
        accumulated_cons_loss = 0.0
        accumulated_ccc_loss = 0.0
        accumulated_spectral_loss = 0.0
        accumulated_gradient_loss = 0.0
        accumulated_spectral_pv_loss = 0.0
        accumulated_mass_loss = 0.0
        num_batches = (num_examples + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            actual_batch_size = min(batch_size, num_examples - batch_idx * batch_size)
            
            # Randomly sample rollout length for this batch
            n_steps = torch.randint(0, current_max_steps + 1, (1,)).item()
            
            # Generate batch of random ICs and convert to normalized grid space
            batch_states = []
            batch_specs = []
            for _ in range(actual_batch_size):
                ic_spec = solver.random_initial_condition(mach=0.2)
                batch_specs.append(ic_spec)
                state = solver.spec2grid(ic_spec)
                state = (state - inp_mean) / torch.sqrt(inp_var)
                batch_states.append(state)
            
            # Stack into batch: (B, C, H, W)
            states = torch.stack(batch_states, dim=0)  # (B, C, H, W) - single timestep state
            
            # --- Noise Injection Logic (Curriculum Rollout) ---
            # For curriculum training, we inject noise at the very beginning (t=0)
            if noise_injection_sigma > 0 and noise_injection_rate > 0 and states.size(0) >= 2:
                batch_size = states.size(0)
                half_batch = batch_size // 2
                
                for i in range(half_batch):
                    if torch.rand(1).item() < noise_injection_rate:
                        target_idx = i + half_batch
                        
                        # Duplicate initial state
                        states[target_idx] = states[i].clone()
                        
                        # Add noise
                        noise = torch.randn_like(states[target_idx]) * noise_injection_sigma
                        states[target_idx] += noise
                        
                        # Note: The ground truth generation (Phase 2) will start from this *noisy* state.
                        # The solver will filter it immediately, so the generated ground truth 
                        # will be clean (filtered) and correct. This is exactly what we want.
            # --------------------------------------------------
            
            # Initialize history buffer for multi-step models
            # Each sample has n_history states, buffer shape: list of (B, C, H, W)
            if n_history > 1:
                history_buffer = [states.clone() for _ in range(n_history)]
            
            # Phase 1: Gradient-free rollout for n_steps (batched)
            with torch.no_grad():
                for _ in range(n_steps):
                    if n_history > 1:
                        # Stack history: (B, n_history*C, H, W)
                        model_input = torch.cat(history_buffer, dim=1)
                    else:
                        model_input = states
                    
                    states = model(model_input)  # Output: (B, C, H, W)
                    
                    # Update history buffer (sliding window)
                    if n_history > 1:
                        history_buffer = history_buffer[1:] + [states.clone()]
            
            # Phase 2: Generate ground truth for each sample
            # (solver operates per-sample, so we still loop here)
            gt_list = []
            for i in range(actual_batch_size):
                # Un-normalize to get physical values
                state_physical = states[i] * torch.sqrt(inp_var.squeeze(0)) + inp_mean.squeeze(0)
                
                # Convert to spectral for solver
                state_spec = solver.grid2spec(state_physical)
                
                # Run solver one step to get ground truth
                gt_spec = solver.timestep(state_spec, nsteps)
                gt_grid = solver.spec2grid(gt_spec)
                gt_normalized = (gt_grid - inp_mean.squeeze(0)) / torch.sqrt(inp_var.squeeze(0))
                gt_list.append(gt_normalized)
            
            # Stack ground truth: (B, C, H, W)
            gt_batch = torch.stack(gt_list, dim=0)
            
            # Phase 3: One ML step WITH gradients (batched)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_mode != "none")):
                # Identify prev_state and prev_prev_state
                prev_state = None
                prev_prev_state = None
                
                if n_history > 1:
                    prev_state = history_buffer[-1]
                    prev_prev_state = history_buffer[-2]
                else:
                    prev_state = states # This is the state used to generate pred
                    prev_prev_state = None # History not tracked in single step curriculum loop easily?
                    # Actually we can track it if we wanted, but let's stick to what's available
                
                # In curriculum/one-step-ahead mode, GT history is identical to Model history
                prev_target = prev_state
                prev_prev_target = prev_prev_state

                pred = model(model_input)
                batch_loss = loss_fn(pred, gt_batch, 
                                     prev_state=prev_state, 
                                     prev_prev_state=prev_prev_state,
                                     prev_target=prev_target,
                                     prev_prev_target=prev_prev_target)
            
            accumulated_loss += batch_loss.item() * actual_batch_size
            # Accumulate loss components if available
            if hasattr(loss_fn, 'last_l2_loss'):
                accumulated_l2_loss += loss_fn.last_l2_loss * actual_batch_size
            if hasattr(loss_fn, 'last_lap_loss'):
                accumulated_lap_loss += loss_fn.last_lap_loss * actual_batch_size
            if hasattr(loss_fn, 'last_cons_loss'):
                accumulated_cons_loss += loss_fn.last_cons_loss * actual_batch_size
            if hasattr(loss_fn, 'last_ccc_loss'):
                accumulated_ccc_loss += loss_fn.last_ccc_loss * actual_batch_size
            if hasattr(loss_fn, 'last_spectral_loss'):
                accumulated_spectral_loss += loss_fn.last_spectral_loss * actual_batch_size
            if hasattr(loss_fn, 'last_gradient_loss'):
                accumulated_gradient_loss += loss_fn.last_gradient_loss * actual_batch_size
            if hasattr(loss_fn, 'last_spectral_pv_loss'):
                accumulated_spectral_pv_loss += loss_fn.last_spectral_pv_loss * actual_batch_size
            if hasattr(loss_fn, 'last_mass_loss'):
                accumulated_mass_loss += loss_fn.last_mass_loss * actual_batch_size
            
            # Backprop
            optimizer.zero_grad(set_to_none=True)
            gscaler.scale(batch_loss).backward()
            gscaler.step(optimizer)
            gscaler.update()
        
        accumulated_loss = accumulated_loss / num_examples
        # Average loss components and store for logging
        if hasattr(loss_fn, 'last_l2_loss'):
            loss_fn.last_l2_loss = accumulated_l2_loss / num_examples
        if hasattr(loss_fn, 'last_lap_loss'):
            loss_fn.last_lap_loss = accumulated_lap_loss / num_examples
        if hasattr(loss_fn, 'last_cons_loss'):
            loss_fn.last_cons_loss = accumulated_cons_loss / num_examples
        if hasattr(loss_fn, 'last_ccc_loss'):
            loss_fn.last_ccc_loss = accumulated_ccc_loss / num_examples
        if hasattr(loss_fn, 'last_spectral_loss'):
            loss_fn.last_spectral_loss = accumulated_spectral_loss / num_examples
        if hasattr(loss_fn, 'last_gradient_loss'):
            loss_fn.last_gradient_loss = accumulated_gradient_loss / num_examples
        if hasattr(loss_fn, 'last_spectral_pv_loss'):
            loss_fn.last_spectral_pv_loss = accumulated_spectral_pv_loss / num_examples
        if hasattr(loss_fn, 'last_mass_loss'):
            loss_fn.last_mass_loss = accumulated_mass_loss / num_examples
        
        if scheduler is not None:
            scheduler.step(accumulated_loss)
        
        epoch_time = time.time() - epoch_start
        
        if logging:
            print(f"--------------------------------------------------------------------------------")
            print(f"Curriculum Epoch {epoch} (max_steps={current_max_steps}):")
            print(f"  time: {epoch_time:.2f}s, loss: {accumulated_loss:.6f}")
            
            # Detailed Loss Breakdown Table (Curriculum)
            if hasattr(loss_fn, 'last_channel_losses'):
                print("\n  Loss Breakdown:")
                print(f"  {'Component':<20} | {'Raw Value':<12} | {'Lambda':<10} | {'Weighted':<12} | {'% Total':<8}")
                print("  " + "-"*75)
                
                loss_components = [
                    ("L2", "last_l2_loss", "l2_lambda"),
                    ("Laplacian", "last_lap_loss", "lap_lambda"),
                    ("Conservation", "last_cons_loss", "cons_lambda"),
                    ("CCC", "last_ccc_loss", "ccc_lambda"),
                    ("Spectral", "last_spectral_loss", "spectral_lambda"),
                    ("Gradient", "last_gradient_loss", "gradient_lambda"),
                    ("UV", "last_uv_loss", "uv_lambda"),
                    ("Spectral PV", "last_spectral_pv_loss", "spectral_pv_lambda"),
                    ("Spectral Mass", "last_mass_loss", "mass_lambda"),
                ]
                
                total_check = 0.0
                component_data = []

                # First pass: collect data and total
                for name, attr_loss, attr_lambda in loss_components:
                    if hasattr(loss_fn, attr_loss) and hasattr(loss_fn, attr_lambda):
                        raw_val = getattr(loss_fn, attr_loss)
                        lam_val = getattr(loss_fn, attr_lambda)
                        if lam_val > 0:
                            weighted_val = raw_val * lam_val
                            component_data.append((name, raw_val, lam_val, weighted_val))
                            total_check += weighted_val
                
                # Second pass: print rows
                for name, raw_val, lam_val, weighted_val in component_data:
                    pct = (weighted_val / total_check * 100) if total_check > 0 else 0
                    print(f"  {name:<20} | {raw_val:<12.6f} | {lam_val:<10.2e} | {weighted_val:<12.6f} | {pct:<6.1f}%")
                print("  " + "-"*75 + "\n")
        
        # Save best checkpoint
        if accumulated_loss < best_loss:
            best_loss = accumulated_loss
            if hasattr(model, 'best_checkpoint_path'):
                if logging:
                    print(f"  New best loss! Saving checkpoint...")
                torch.save(model.state_dict(), model.best_checkpoint_path)
        
        # Always save recent checkpoint (every epoch)
        if hasattr(model, 'recent_checkpoint_path'):
            torch.save(model.state_dict(), model.recent_checkpoint_path)
            
        # Save a copy to scratch directory if provided
        if scratch_dir is not None:
            if not os.path.exists(scratch_dir):
                os.makedirs(scratch_dir, exist_ok=True)
            scratch_checkpoint_path = os.path.join(scratch_dir, f"curriculum_checkpoint_{epoch}.pt")
            torch.save(model.state_dict(), scratch_checkpoint_path)

        
        # Periodic plot logging (commit=False, so plots merge into the metrics log below)
        if plot_callback is not None:
            plot_callback(model, epoch_label=f"curriculum_ep{epoch}")

        # Log to wandb
        if logging and wandb is not None and wandb.run is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            log_dict = {
                "curriculum_loss": accumulated_loss,
                "curriculum_max_steps": current_max_steps,
                "curriculum_lr": current_lr,
            }
            # Log separate loss components if available
            if hasattr(loss_fn, 'last_l2_loss'):
                log_dict["L2 loss"] = loss_fn.last_l2_loss
            if hasattr(loss_fn, 'last_lap_loss'):
                log_dict["Laplacian loss"] = loss_fn.last_lap_loss
            if hasattr(loss_fn, 'last_cons_loss'):
                log_dict["Conservation loss"] = loss_fn.last_cons_loss
            if hasattr(loss_fn, 'last_ccc_loss'):
                log_dict["CCC loss"] = loss_fn.last_ccc_loss
            if hasattr(loss_fn, 'last_spectral_loss'):
                log_dict["Spectral loss"] = loss_fn.last_spectral_loss
            if hasattr(loss_fn, 'last_gradient_loss'):
                log_dict["Gradient loss"] = loss_fn.last_gradient_loss
            if hasattr(loss_fn, 'last_uv_loss'):
                log_dict["UV loss"] = loss_fn.last_uv_loss
            if hasattr(loss_fn, 'last_spectral_pv_loss'):
                log_dict["Spectral PV loss"] = loss_fn.last_spectral_pv_loss
            if hasattr(loss_fn, 'last_mass_loss'):
                log_dict["Spectral Mass loss"] = loss_fn.last_mass_loss
            wandb.log(log_dict)
    
    train_time = time.time() - train_start
    print(f"--------------------------------------------------------------------------------")
    print(f"Curriculum training done. Took {train_time:.2f}s")
    return accumulated_loss


def parse_list_arg(value, dtype=int):
    """Parse a comma-separated string or single value into a list.
    
    Args:
        value: Either a single value, or a comma-separated string (e.g., "12,6,6")
        dtype: Type to convert each element to (int or float)
    
    Returns:
        List of values of the specified type
    """
    if isinstance(value, str):
        return [dtype(x.strip()) for x in value.split(",")]
    else:
        return [dtype(value)]


def main(root_path, pretrain_epochs=100, finetune_epochs=10, batch_size=1, learning_rate=1e-3, train=True, load_checkpoint=False, checkpoint_name="checkpoint.pt", amp_mode="none", log_grads=0, validate_steps=96, nics=50, finetune_steps=2, curriculum_epochs=0, curriculum_max_steps=None, seed=None, l2_lambda=1.0, laplacian_lambda=0.0, conservation_lambda=0.0, ccc_lambda=0.0, spectral_lambda=0.0, gradient_lambda=0.0, vorticity_lambda=1.0, uv_lambda=0.0, spectral_pv_lambda=0.0, mass_lambda=0.0, time_diff2_lambda=0.0, run_tag=None, num_examples=256, ic_mach=0.2, ic_llimit=25, stochastic_ic_llimit=False, stochastic_ic_mach=False, plotting_dt=0, ic_spinup_max=0, n_history=1, precomputed_db_path=None, first_step_lambda=1.0, first_step_only_l2_lap_uv=False, checkpoint_activation=False):

    # enable logging by default
    logging = True

    # set seed (random if not specified)
    if seed is None:
        seed = torch.randint(0, 2**31, (1,)).item()
    print(f"Using random seed: {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device.index)

    # 30 minute prediction steps. note: the attention models have "stability
    # issues" for dt=3600s, so we lower to 1800s. (solution falls apart using
    # autoregressive rollout) LSNO and SFNO are stable at 3600s.
    dt = 600# 1800 #1 * 1800
    dt_solver = 150
    nsteps = dt // dt_solver
    grid = "equiangular"
    nlat, nlon = (192, 288)
    # dt = 3600 #1 * 1800
    # dt_solver = 150
    # nsteps = dt // dt_solver
    # grid = "equiangular"
    # nlat, nlon = (256, 512)
    dataset = PdeDatasetExtended(
        dt=dt, nsteps=nsteps, dims=(nlat, nlon), device=device, grid=grid, normalize=True, nfuture=0,
        ic_mach=ic_mach,
        ic_llimit=ic_llimit,
        stochastic_ic_llimit=stochastic_ic_llimit,
        stochastic_ic_mach=stochastic_ic_mach,
        ic_spinup_max=ic_spinup_max,
        num_examples=num_examples
    )
    dataset.sht = RealSHT(nlat=nlat, nlon=nlon, grid=grid).to(device=device)
    
    # Override single-sample normalization with robust equilibrium stats
    mach = ic_mach  # Match the mach number used in PdeDataset._get_sample()
    norm_stats = get_or_compute_stats(
        solver=dataset.solver,
        cache_dir=root_path,
        n_samples=1000,      # Number of equilibrated samples for statistics
        mach=mach,
        spinup_steps=50,    # Run solver 100 steps to reach equilibrium
        dt=dt,
        dt_solver=dt_solver,
        show_progress=True
    )
    dataset.inp_mean = norm_stats["mean"]
    dataset.inp_var = norm_stats["var"]
    
    # Compute global U and V standard deviations separately for loss normalization
    # uv_var is (2, 1, 1), indices 0=u, 1=v
    if "uv_var" in norm_stats:
        uv_var = norm_stats["uv_var"]
        u_std = torch.sqrt(uv_var[0]).mean().item()
        v_std = torch.sqrt(uv_var[1]).mean().item()
    else:
        # Fallback for old stats files
        print("Warning: 'uv_var' key missing in stats file. Falling back to global var.")
        u_std = torch.sqrt(dataset.inp_var.mean()).item() 
        v_std = torch.sqrt(dataset.inp_var.mean()).item()
        
    print(f"Computed UV normalization stds: U={u_std:.2f}, V={v_std:.2f} m/s")
    
    precomputed_ds = None
    # Conditional training DataLoader: precomputed Zarr DB or on-the-fly solver
    if args.precomputed_db:
        from precomputed_dataset import PrecomputedTrajectoryDataset
        zarr_path = os.path.join(args.precomputed_db, "trajectories.zarr")
        stats_path = os.path.join(args.precomputed_db, "stats.npz")
        precomputed_ds = PrecomputedTrajectoryDataset(
            zarr_path=zarr_path,
            stats_path=stats_path,
            n_history=n_history,
            nfuture=0,
            normalize=True,
        )
        # Zarr is process-safe, so we can use multi-worker loading
        # Lowering to 4 workers to avoid saturating 8 CPU allocation
        # Use RandomSampler to limit epoch size to num_examples (not whole dataset).
        #
        # IMPORTANT: We must rebuild the DataLoader (not just mutate dataset.nfuture)
        # because persistent_workers=True forks workers once on first iteration and
        # each worker holds its own frozen copy of the dataset object. Setting
        # dataset.nfuture on the main process never reaches the workers, so they
        # would keep serving single-step targets even during autoregressive finetuning.
        def _make_precomputed_loader(nfuture_val):
            """Rebuild the DataLoader so workers see the updated nfuture window."""
            precomputed_ds.nfuture = nfuture_val  # rebuilds self.index in-place
            sampler = RandomSampler(precomputed_ds, replacement=True,
                                    num_samples=num_examples)
            return DataLoader(precomputed_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=14,
                              prefetch_factor=16, persistent_workers=True,
                              pin_memory=True)

        dataloader = _make_precomputed_loader(nfuture_val=0)
        print(f"Using pre-computed training data from: {args.precomputed_db}")
        print(f"  {len(precomputed_ds)} total samples. Sampling {num_examples} per epoch.")
        print("  Starting training loop...")

        # Override u_std/v_std from the DB's own stats.npz so the UV loss is
        # normalized against the actual distribution of the training data.
        db_stats = np.load(stats_path)
        if "uv_var" in db_stats and not np.allclose(db_stats["uv_var"], 0.0):
            uv_var_db = torch.from_numpy(db_stats["uv_var"])
            u_std = torch.sqrt(uv_var_db[0]).mean().item()
            v_std = torch.sqrt(uv_var_db[1]).mean().item()
            print(f"  UV stds from DB stats.npz: U={u_std:.2f}, V={v_std:.2f} m/s")
        else:
            print("  WARNING: uv_var missing or zero in stats.npz — re-run generate_training_db.py "
                  "without --skip_stats, or run compute_db_stats.py to add UV stats.")
    else:
        # On-the-fly solver generation (original path)
        # There is still an issue with parallel dataloading. Do NOT use it at the moment
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, persistent_workers=False)


    nlat = dataset.nlat
    nlon = dataset.nlon

    # prepare dicts containing models and corresponding metrics
    models = {}

    # Calculate input channels based on n_history
    # 3 channels per timestep (height, vorticity, divergence)
    in_chans = 3 * n_history

    # get baseline model registry
    baseline_models = get_baseline_models(img_size=(nlat, nlon), in_chans=in_chans, out_chans=3, residual_prediction=True, grid=grid, checkpoint_activation=checkpoint_activation)

    # specify which models to train here
    # Format: "Run_Name": baseline_models["Architecture_Name"]
    models = {
        #  "sfno_sc2_layers4_e32": baseline_models["sfno_sc2_layers4_e32"],
        #  "sfno_sc2_layers4_e128_FR": baseline_models["sfno_sc2_layers4_e128"],
        #  "nnorm_gam29_d600": baseline_models["nnorm_gam2"],
        #  "disco_mlp_transformer_e64": baseline_models["disco_mlp_transformer_e64"],
        #  "EPD_W9_MB4_D32": baseline_models["encode_process_decode"],
        #  "EPD_W9_MB1_D66_L": baseline_models["EPD_W9_MB1_D66_L"],
        #  "disco_epd_W9_MB1_D66_L": baseline_models["disco_epd_W9_MB1_D66_L"],
        # "disco_epd_W9_MB1_D66_L_morlet": baseline_models["disco_epd_W9_MB1_D66_L_morlet"],
        #  "hierarchical_test_5x5": baseline_models["hierarchical_disco_epd"]
        # "sfno_sc2_layers4_e32_T1": baseline_models["sfno_sc2_layers4_e32"],
        # "film_filtered_diffusion_AR6_2": baseline_models["film_filtered_diffusion"],
        # "film_curriculum_rollout_T2": baseline_models["film"],
        # "film_filtered_diffusion_curriculum_ro
        # llout_T2": baseline_models["film_filtered_diffusion"],
        # "film_filtered_diffusion_curriculum_reg_sched": baseline_models["film_filtered_diffusion"],
        # "film_laplacian_8": baseline_models["film"],
        # "film_laplacian_8": baseline_models["film_conservation"],
        # "film_L7_MC5_H3": baseline_models["film_history3"],
        # "film_history3_conservation_CCC7_l23": baseline_models["film_history3_conservation"],
        # "film_conservation_CCC0_l21": baseline_models["film_conservation"],
        # "noise_T1_stochastic_lic_half": baseline_models["film_conservation"],
        # "noise_T1_stochastic_lic_half2": baseline_models["film_conservation"],
        # "plot_testing_icsu100_noise_laplacian": baseline_models["film_conservation"],
        # "film_conservation_precomputed_ctrl": baseline_models["film_conservation"],
        # "film_conservation_spectral_norm": baseline_models["film_conservation_spectral_norm"],
        # "s2ntransformer_lap8_ccc1_l23": baseline_models["s2ntransformer_sc2_layers4_e128"],
        # "noise_laplacian_history3": baseline_models["film_conservation_history3"],
        # "mlp_transformer_e64_h2_w9_spectral": baseline_models["mlp_transformer_e64_h2"],
        # "mlp_transformer_e126_h4_control": baseline_models["mlp_transformer_e126_h4"],
        # "disco_mlp_transformer_e64_control": baseline_models["disco_mlp_transformer_e64"],
        # "mlp_transformer_e126_h4_no_lat_control": baseline_models["mlp_transformer_e126_h4_no_lat"],
        # "s2ntransformer_sc2_layers4_e128_control2": baseline_models["s2ntransformer_sc2_layers4_e128"],
        # "s2ntransformer_sc2_layers4_e128_k55_tc9": baseline_models["s2ntransformer_sc2_layers4_e128_k55_tc9"],
        # "s2unet_sc2_layers4_e128_morlet_L2_only": baseline_models["s2unet_sc2_layers4_e128"],
        # "s2ntransformer_precomputed_ctrl": baseline_models["s2ntransformer_sc2_layers4_e128"],
        # "s2ntransformer_sc2_layers4_e128_L2_NS8192_Lap8Noise": baseline_models["s2ntransformer_sc2_layers4_e128"],
         "sfno_sc2_layers4_e128_L2_NS8192_FSL_T1": baseline_models["sfno_sc2_layers4_e128"],
        #  "sfno_sc3_layers4_e128_L2": baseline_models["sfno_sc3_layers4_e128"],
        #  "lsno_sc2_layers4_e128_NS8192_Lap8": baseline_models["lsno_sc2_layers4_e128"],




    }

    # project_str = "cons_and_attn"
    # project_str = "time_laplacian"
    project_str = "precomputed_2"
    # project_str = "mlp_transformer_testing"

    # Available Architectures in baseline_models:
    # "s2transformer_sc2_layers4_e128", # Spherical Transformer (Global Attention)
    # "s2ntransformer_sc2_layers4_e128",# Spherical Transformer (Neighborhood Attention)
    # "s2segformer_sc2_layers4_e128",   # Spherical Segformer (Global Attention)
    # "s2nsegformer_sc2_layers4_e128",  # Spherical Segformer (Neighborhood Attention)
    # "sfno_sc2_layers4_e32",           # SFNO
    # "lsno_sc2_layers4_e32",           # LSNO

    # loss function
    l2_loss_fn = SquaredL2LossS2(nlat=nlat, nlon=nlon, grid=grid).to(device)
    
    # optional loss components
    lap_loss_fn = LaplacianLossS2(nlat=nlat, nlon=nlon, grid=grid).to(device) if laplacian_lambda > 0 else None
    cons_loss_fn = ConservationLossS2(nlat=nlat, nlon=nlon, grid=grid).to(device) if conservation_lambda > 0 else None
    spectral_loss_fn = SpectralLossS2(nlat=nlat, nlon=nlon, grid=grid, lmin_loss=args.spectral_lmin, lmax_loss=args.spectral_lmax, decay_exponent=args.spectral_decay).to(device) if args.spectral_lambda > 0 else None
    gradient_loss_fn = GradientLossS2(nlat=nlat, nlon=nlon, grid=grid).to(device) if args.gradient_lambda > 0 else None
    
    # Wrapper class to track all loss components
    class CombinedLoss(nn.Module):
        CHANNEL_NAMES = ["height", "vorticity", "divergence"]
        
        
        def __init__(self, l2_fn, lap_fn=None, cons_fn=None, ccc_fn=None, spectral_fn=None, gradient_fn=None,
                 l2_lambda=1.0, lap_lambda=0.0, cons_lambda=0.0, ccc_lambda=0.0, spectral_lambda=0.0, gradient_lambda=0.0, 
                 vorticity_lambda=1.0, uv_lambda=0.0, spectral_pv_lambda=0.0, mass_lambda=0.0, time_diff2_lambda=0.0,
                 first_step_lambda=1.0, first_step_only_l2_lap_uv=False,
                 solver=None, inp_mean=None, inp_var=None, u_std=None, v_std=None, device=None):
            super().__init__()
            self.l2_fn = l2_fn
            self.lap_fn = lap_fn
            self.cons_fn = cons_fn
            self.ccc_fn = ccc_fn
            self.spectral_fn = spectral_fn
            self.gradient_fn = gradient_fn
            self.l2_lambda = l2_lambda
            self.lap_lambda = lap_lambda
            self.cons_lambda = cons_lambda
            self.ccc_lambda = ccc_lambda
            self.spectral_lambda = spectral_lambda
            self.gradient_lambda = gradient_lambda
            self.vorticity_lambda = vorticity_lambda
            self.uv_lambda = uv_lambda
            self.spectral_pv_lambda = spectral_pv_lambda
            self.mass_lambda = mass_lambda
            self.time_diff2_lambda = time_diff2_lambda
            self.first_step_lambda = first_step_lambda
            self.first_step_only_l2_lap_uv = first_step_only_l2_lap_uv
            
            # UV Reconstruction Loss params and Spectral PV/Mass Loss params
            self.solver = solver
            self.inp_mean = inp_mean
            self.inp_var = inp_var
            self.u_std = u_std
            self.v_std = v_std
            
            # Initialize Spectral PV Loss if enabled
            self.pv_loss_fn = None
            if self.spectral_pv_lambda > 0 and self.solver is not None and TwoStepPVResidual is not None:
                 # Check for cached lats in solver
                if hasattr(self.solver, "sht") and hasattr(self.solver.sht, "lats"):
                     # solver.sht.lats are usually colatitudes in radians (0 to pi)
                     colat_rad = self.solver.sht.lats
                     lat_deg = 90.0 - torch.rad2deg(colat_rad)
                else:
                    lat_deg = np.linspace(-90, 90, self.solver.nlat)
                
                if device is None:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                self.pv_loss_fn = TwoStepPVResidual(
                    lats_deg=lat_deg,
                    dt_seconds=600.0, # Fixed DT for this setup
                    omega=getattr(self.solver, 'omega', 7.292e-5),
                    radius=getattr(self.solver, 'radius', 6.37122e6),
                    gravity=getattr(self.solver, 'gravity', 9.80616),
                    mode="pv_advective",
                    height_mode="geopotential" # Data is geopotential
                ).to(device=device)

            # Initialize Spectral Mass Loss if enabled
            self.mass_loss_fn = None
            if self.mass_lambda > 0 and self.solver is not None and TwoStepMassResidual is not None:
                 # Check for cached lats in solver
                if hasattr(self.solver, "sht") and hasattr(self.solver.sht, "lats"):
                     # solver.sht.lats are usually colatitudes in radians (0 to pi)
                     colat_rad = self.solver.sht.lats
                     lat_deg = 90.0 - torch.rad2deg(colat_rad)
                else:
                    lat_deg = np.linspace(-90, 90, self.solver.nlat)
                
                if device is None:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                self.mass_loss_fn = TwoStepMassResidual(
                    lats_deg=lat_deg,
                    dt_seconds=600.0, # Fixed DT for this setup
                    height_mode="geopotential" # Data is geopotential
                ).to(device=device)
            
            # Store accumulated values for logging
            self.last_l2_loss = 0.0
            self.last_lap_loss = 0.0
            self.last_cons_loss = 0.0
            self.last_ccc_loss = 0.0
            self.last_spectral_loss = 0.0
            self.last_gradient_loss = 0.0
            self.last_uv_loss = 0.0
            self.last_spectral_pv_loss = 0.0
            self.last_mass_loss = 0.0
            self.last_time_diff2_loss = 0.0
            self.last_channel_losses = {name: 0.0 for name in self.CHANNEL_NAMES}
            
        def reset_stats(self):
            """Reset the step accumulations at the beginning of a rollout."""
            self.last_l2_loss = 0.0
            self.last_lap_loss = 0.0
            self.last_cons_loss = 0.0
            self.last_ccc_loss = 0.0
            self.last_spectral_loss = 0.0
            self.last_gradient_loss = 0.0
            self.last_uv_loss = 0.0
            self.last_spectral_pv_loss = 0.0
            self.last_mass_loss = 0.0
            self.last_time_diff2_loss = 0.0
            for k in self.last_channel_losses:
                self.last_channel_losses[k] = 0.0
        
        def forward(self, pred, target, prev_state=None, prev_prev_state=None, prev_target=None, prev_prev_target=None, step=0):
            """Compute combined loss with optional vorticity weighting."""
            total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            
            # Determine effective lambdas for the current step
            eff_l2_lambda = self.l2_lambda
            eff_lap_lambda = self.lap_lambda
            eff_uv_lambda = self.uv_lambda
            
            mask = 1.0
            if self.first_step_only_l2_lap_uv and step > 0:
                eff_l2_lambda = 0.0
                eff_lap_lambda = 0.0
                eff_uv_lambda = 0.0
                mask = 0.0
                
            step_mult = self.first_step_lambda if step == 0 else 1.0
                
            # L2 Loss (with optional vorticity weighting)
            if eff_l2_lambda > 0:
                # Compute per-channel L2 losses
                channel_losses = []
                for c, name in enumerate(self.CHANNEL_NAMES):
                    if c < pred.shape[1]:  # Safety check
                        ch_loss = self.l2_fn(pred[:, c:c+1], target[:, c:c+1])
                        self.last_channel_losses[name] = ch_loss.item()
                        channel_losses.append(ch_loss)
                
                # Apply vorticity weighting (channel index 1)
                if self.vorticity_lambda != 1.0 and len(channel_losses) > 1:
                    # Simple weighted sum: height + vorticity_lambda*vorticity + divergence
                    l2 = channel_losses[0] + self.vorticity_lambda * channel_losses[1] + channel_losses[2]
                else:
                    # Standard L2 (all channels weighted equally)
                    l2 = self.l2_fn(pred, target)
                
                self.last_l2_loss += l2.item() * step_mult * mask
                total = total + eff_l2_lambda * l2
            
            # Laplacian Loss
            if self.lap_fn is not None and eff_lap_lambda > 0:
                lap = self.lap_fn(pred, target)
                self.last_lap_loss += lap.item() * step_mult * mask
                total = total + eff_lap_lambda * lap
            
            # Conservation Loss
            if self.cons_fn is not None and self.cons_lambda > 0:
                cons = self.cons_fn(pred, target)
                self.last_cons_loss += cons.item() * step_mult
                total = total + self.cons_lambda * cons
            
            # CCC Loss (needs channel-last format)
            if self.ccc_fn is not None and self.ccc_lambda > 0:
                # CCC expects (B, H, W, C), we have (B, C, H, W)
                pred_nhwc = pred.permute(0, 2, 3, 1)
                target_nhwc = target.permute(0, 2, 3, 1)
                ccc = self.ccc_fn(pred_nhwc, target_nhwc)
                self.last_ccc_loss += ccc.item() * step_mult
                total = total + self.ccc_lambda * ccc
                
            # Spectral Loss
            if self.spectral_fn is not None and self.spectral_lambda > 0:
                spec = self.spectral_fn(pred, target)
                self.last_spectral_loss += spec.item() * step_mult
                total = total + self.spectral_lambda * spec
                
            # Gradient Loss
            if self.gradient_fn is not None and self.gradient_lambda > 0:
                grad = self.gradient_fn(pred, target)
                self.last_gradient_loss += grad.item() * step_mult
                total = total + self.gradient_lambda * grad
                
            # UV Reconstruction Loss
            if eff_uv_lambda > 0 and self.solver is not None:
                # 1. Denormalize to get physical grid values
                # pred/target: (B, C, H, W)
                pred_phys = pred * torch.sqrt(self.inp_var) + self.inp_mean
                target_phys = target * torch.sqrt(self.inp_var) + self.inp_mean
                
                # 2. Convert to spectral space
                # grid2spec expects (C, H, W), so we iterate over batch
                # Typically batch_size is small (1) in this script
                uv_loss = 0.0
                bs = pred.shape[0]
                for i in range(bs):
                    pred_spec = self.solver.grid2spec(pred_phys[i])
                    target_spec = self.solver.grid2spec(target_phys[i])
                    
                    # 3. Compute U/V from vorticity/divergence (indices 1:)
                    # getuv returns (2, H, W) -> u, v
                    pred_uv = self.solver.getuv(pred_spec[1:])     # (2, H, W)
                    target_uv = self.solver.getuv(target_spec[1:]) # (2, H, W)
                    
                    # 4. Normalize separately by u_std and v_std
                    # u component (index 0)
                    diff_u = (pred_uv[0] - target_uv[0]) / self.u_std
                    # v component (index 1)
                    diff_v = (pred_uv[1] - target_uv[1]) / self.v_std
                    
                    # MSE (sum of squares of both components)
                    uv_loss += (diff_u ** 2).mean() + (diff_v ** 2).mean()
                
                # Average over batch
                uv_loss = uv_loss / bs
                
                self.last_uv_loss += uv_loss.item() * step_mult * mask
                total = total + eff_uv_lambda * uv_loss
            
            # Spectral PV Loss
            if self.spectral_pv_lambda > 0 and self.pv_loss_fn is not None and prev_state is not None:
                # 1. Prepare states
                # prev_state might be stacked (B, n*C, H, W). We need the last state (B, C, H, W).
                C = pred.shape[1]
                if prev_state.shape[1] > C:
                    state0_norm = prev_state[:, -C:]
                else:
                    state0_norm = prev_state
                state1_norm = pred

                # 2. Denormalize to physical units (PV loss expects physical)
                st0 = state0_norm * torch.sqrt(self.inp_var) + self.inp_mean
                st1 = state1_norm * torch.sqrt(self.inp_var) + self.inp_mean
                
                # 3. Compute U, V, Zeta, H for both steps
                # state: channels 0=h, 1=z, 2=d
                
                pv_loss = 0.0
                bs = pred.shape[0]
                for i in range(bs):
                    # Step 0
                    sp0 = self.solver.grid2spec(st0[i])
                    uv0 = self.solver.getuv(sp0[1:]) # (2, H, W)
                    u0, v0 = uv0[0], uv0[1]
                    z0 = st0[i, 1]
                    h0 = st0[i, 0]
                    
                    # Step 1
                    sp1 = self.solver.grid2spec(st1[i])
                    uv1 = self.solver.getuv(sp1[1:])
                    u1, v1 = uv1[0], uv1[1]
                    z1 = st1[i, 1]
                    h1 = st1[i, 0]
                    
                    loss_i, _, _ = self.pv_loss_fn(
                        u0.unsqueeze(0), v0.unsqueeze(0), z0.unsqueeze(0), None, h0.unsqueeze(0),
                        u1.unsqueeze(0), v1.unsqueeze(0), z1.unsqueeze(0), None, h1.unsqueeze(0)
                    )
                    pv_loss += loss_i
                
                pv_loss = pv_loss / bs
                self.last_spectral_pv_loss += pv_loss.item() * step_mult
                total = total + self.spectral_pv_lambda * pv_loss

            # Spectral Mass Loss
            if self.mass_lambda > 0 and self.mass_loss_fn is not None and prev_state is not None:
                # 1. Prepare states
                C = pred.shape[1]
                if prev_state.shape[1] > C:
                    state0_norm = prev_state[:, -C:]
                else:
                    state0_norm = prev_state
                state1_norm = pred

                # 2. Denormalize to physical units
                st0 = state0_norm * torch.sqrt(self.inp_var) + self.inp_mean
                st1 = state1_norm * torch.sqrt(self.inp_var) + self.inp_mean
                
                mass_loss = 0.0
                bs = pred.shape[0]
                for i in range(bs):
                    # Step 0
                    sp0 = self.solver.grid2spec(st0[i])
                    uv0 = self.solver.getuv(sp0[1:]) # (2, H, W)
                    u0, v0 = uv0[0], uv0[1]
                    h0 = st0[i, 0]
                    
                    # Step 1
                    sp1 = self.solver.grid2spec(st1[i])
                    uv1 = self.solver.getuv(sp1[1:])
                    u1, v1 = uv1[0], uv1[1]
                    h1 = st1[i, 0]
                    
                    loss_i, _, _ = self.mass_loss_fn(
                        u0.unsqueeze(0), v0.unsqueeze(0), h0.unsqueeze(0),
                        u1.unsqueeze(0), v1.unsqueeze(0), h1.unsqueeze(0)
                    )
                    mass_loss += loss_i
                
                mass_loss = mass_loss / bs
                self.last_mass_loss += mass_loss.item() * step_mult
                total = total + self.mass_lambda * mass_loss

            # Time derivative second difference loss
            if self.time_diff2_lambda > 0 and prev_state is not None and prev_prev_state is not None and prev_target is not None and prev_prev_target is not None:
                # Extract last frame filters are needed if history is stacked (n_history > 1)
                
                # Handling stacked history (B, n*C, H, W) vs single (B, C, H, W)
                C = pred.shape[1]
                
                # Helper to extract last step from history stack
                def extract_last(state):
                    if state.shape[1] > C:
                        return state[:, -C:]
                    return state

                curr = pred
                prev = extract_last(prev_state)
                prev_prev = extract_last(prev_prev_state)
                
                curr_tar = target
                prev_tar = extract_last(prev_target)
                prev_prev_tar = extract_last(prev_prev_target)

                # Estimated Second Time Derivatives
                # u_{t} - 2u_{t-1} + u_{t-2}
                diff2_pred = curr - 2 * prev + prev_prev
                diff2_gt = curr_tar - 2 * prev_tar + prev_prev_tar
                
                # Match the acceleration
                loss_diff2 = self.l2_fn(diff2_pred, diff2_gt)
                
                self.last_time_diff2_loss += loss_diff2.item() * step_mult
                total = total + self.time_diff2_lambda * loss_diff2

            if step == 0:
                total = total * self.first_step_lambda
                
            return total
    
    # Initialize CCC loss if needed
    ccc_loss_fn = CCCLoss().to(device) if ccc_lambda > 0 else None
    
    loss_fn = CombinedLoss(
        l2_fn=l2_loss_fn, 
        lap_fn=lap_loss_fn, 
        cons_fn=cons_loss_fn,
        ccc_fn=ccc_loss_fn,
        spectral_fn=spectral_loss_fn,
        gradient_fn=gradient_loss_fn,
        l2_lambda=l2_lambda, 
        lap_lambda=laplacian_lambda,
        cons_lambda=conservation_lambda,
        ccc_lambda=ccc_lambda,
        spectral_lambda=args.spectral_lambda,
        gradient_lambda=args.gradient_lambda,
        vorticity_lambda=vorticity_lambda,
        uv_lambda=uv_lambda,
        spectral_pv_lambda=args.spectral_pv_lambda,
        mass_lambda=args.mass_lambda,
        time_diff2_lambda=args.time_diff2_lambda,
        first_step_lambda=first_step_lambda,
        first_step_only_l2_lap_uv=first_step_only_l2_lap_uv,
        solver=dataset.solver,
        inp_mean=dataset.inp_mean,
        inp_var=dataset.inp_var,
        u_std=u_std,
        v_std=v_std,
        device=device
    ).to(device)
    
    loss_desc = f"{l2_lambda} * L2"
    if vorticity_lambda != 1.0:
        loss_desc += f" (vorticity weight={vorticity_lambda})"
    if laplacian_lambda > 0:
        loss_desc += f" + {laplacian_lambda} * Laplacian"
    if conservation_lambda > 0:
        loss_desc += f" + {conservation_lambda} * Conservation"
    if ccc_lambda > 0:
        loss_desc += f" + {ccc_lambda} * CCC"
    if args.spectral_lambda > 0:
        loss_desc += f" + {args.spectral_lambda} * Spectral (l>{args.spectral_lmin}, k={args.spectral_decay})"
    if args.gradient_lambda > 0:
        loss_desc += f" + {args.gradient_lambda} * Gradient"
    if uv_lambda > 0:
        loss_desc += f" + {uv_lambda} * UV_Recon"
    if args.spectral_pv_lambda > 0:
        loss_desc += f" + {args.spectral_pv_lambda} * Spectral_PV"
    if args.mass_lambda > 0:
        loss_desc += f" + {args.mass_lambda} * Spectral_Mass"
    if args.time_diff2_lambda > 0:
        loss_desc += f" + {args.time_diff2_lambda} * Time_Diff2"
    print(f"Using loss: {loss_desc}")

    # dictionary for logging the metrics
    metrics = {}
    metrics_fns = {
        "L2 error": L2LossS2(nlat=nlat, nlon=nlon, grid=grid).to(device=device),
        "L1 error": L1LossS2(nlat=nlat, nlon=nlon, grid=grid).to(device=device),
        "W11 error": W11LossS2(nlat=nlat, nlon=nlon, grid=grid).to(device=device),
    }

    # iterate over models and train each model
    for model_name, model_handle in models.items():

        # Initialize wandb at the start of each model to capture entire log
        if logging and wandb is not None and train:
            wandb_kwargs = dict(
                project=project_str,
                group=model_name,
                name=model_name + "_" + str(time.time()),
                config={**model_handle.keywords, "PBS_JOBID": PBS_JOBID},
                settings=wandb.Settings(console="wrap"),
            )
            if args.wandb_run_id:
                wandb_kwargs["id"] = args.wandb_run_id
                wandb_kwargs["resume"] = "must"
            run = wandb.init(**wandb_kwargs)
            print(f"PBS Job ID: {PBS_JOBID}", flush=True)
        else:
            run = None

        model = model_handle().to(device)
        
        # Check if model has a fixed n_history preference (from registry)
        # If not, apply the command-line argument
        if not hasattr(model, 'n_history'):
            model.n_history = n_history
        
        # Get n_history from model attribute (now guaranteed to exist)
        current_n_history = getattr(model, 'n_history', 1)
        
        if current_n_history > 1:
            print(f"Multi-history model detected: n_history={current_n_history}")
            # Update dataset with correct n_history for training data
            dataset.n_history = current_n_history
        else:
             # Ensure dataset uses 1 if model is single-step
             dataset.n_history = 1

        print(model)

        metrics[model_name] = {}

        num_params = count_parameters(model)
        print(f"number of trainable params: {num_params}")
        metrics[model_name]["num_params"] = num_params

        exp_dir = os.path.join(root_path, model_name)
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir, exist_ok=True)
            
        # Attach best checkpoint path to model (derive from checkpoint_name to preserve lineages)
        if checkpoint_name and checkpoint_name != "checkpoint.pt":
            base_name = os.path.splitext(os.path.basename(checkpoint_name))[0]
            best_name = f"best_{base_name}.pt"
            recent_name = f"recent_{base_name}.pt"
        else:
            best_name = "best_checkpoint.pt"
            recent_name = "recent_checkpoint.pt"
        model.best_checkpoint_path = os.path.join(exp_dir, best_name)
        model.recent_checkpoint_path = os.path.join(exp_dir, recent_name)

        if load_checkpoint:
            print(f"Loading checkpoint: {checkpoint_name}")
            if device.type == "cpu":
                model.load_state_dict(torch.load(os.path.join(exp_dir, checkpoint_name), map_location=torch.device("cpu")), strict=False)
            else:
                model.load_state_dict(torch.load(os.path.join(exp_dir, checkpoint_name)), strict=False)

        # run the training
        if train:
            # Setup scratch directory for checkpoints, e.g. /glade/derecho/scratch/$USER/TH_SWE_output/model_name/checkpoints
            scratch_root = f"/glade/derecho/scratch/{os.environ['USER']}/TH_SWE_output"
            model_scratch_dir = os.path.join(scratch_root, model_name, "checkpoints")
            if not os.path.exists(model_scratch_dir):
                os.makedirs(model_scratch_dir, exist_ok=True)
            if logging:
                print(f"Saving per-epoch checkpoints to: {model_scratch_dir}")

            # Parse learning rate - use first value for pretraining
            base_learning_rate = parse_list_arg(learning_rate, dtype=float)[0]
            optimizer = torch.optim.Adam(model.parameters(), lr=base_learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
            gscaler = torch.GradScaler("cuda", enabled=(amp_mode == "fp16"))

            start_time = time.time()

            if logging:
                print(f"Training {model_name}, single step")

            # ── Periodic plot callback ──────────────────────────────────────
            plot_callback = None
            if plotting_dt > 0 and wandb is not None:
                _last_plot_time = [0.0]  # mutable container for closure
                _plot_nics = 3
                _plot_steps = min(92, validate_steps)  # shorter than full validation
                _dt_minutes = (dt_solver * nsteps) / 60.0  # minutes per ML step

                def generate_and_log_plots(model, epoch_label=""):
                    """Run a quick rollout and log plots to wandb (if enough time has passed)."""
                    now = time.time()
                    if now - _last_plot_time[0] < plotting_dt:
                        return  # too soon
                    _last_plot_time[0] = now

                    model.eval()
                    print(f"  [plot_callback] Generating plots ({_plot_nics} ICs, {_plot_steps} steps)...")
                    plot_start = time.time()

                    try:
                        # Quick rollout (same logic as autoregressive_inference, but lighter)
                        all_preds = []
                        all_truth = []
                        inp_mean = dataset.inp_mean
                        inp_var = dataset.inp_var

                        with torch.inference_mode():
                            for iic in range(_plot_nics):
                                use_precomputed_truth = False
                                states_chunk = None

                                if iic == 0:
                                    ic = dataset.solver.galewsky_initial_condition()
                                else:
                                    if precomputed_ds is not None:
                                        use_precomputed_truth = True
                                        max_start = precomputed_ds.T_total - 1 - _plot_steps
                                        if max_start < 0:
                                             # Should not happen if _plot_steps is reasonable
                                             ic = dataset.solver.random_initial_condition(mach=ic_mach)
                                             use_precomputed_truth = False
                                        else:
                                            traj_idx = torch.randint(0, precomputed_ds.N, (1,)).item()
                                            t_idx = torch.randint(0, max_start + 1, (1,)).item()
                                            states_chunk = torch.from_numpy(precomputed_ds.states[traj_idx, t_idx : t_idx + _plot_steps + 1])
                                            states_chunk = states_chunk.float().to(device)
                                            state0_grid = states_chunk[0]
                                    else:
                                        ic = dataset.solver.random_initial_condition(mach=ic_mach)

                                if not use_precomputed_truth:
                                    state0_grid = dataset.solver.spec2grid(ic)
                                    
                                state0 = (state0_grid - inp_mean) / torch.sqrt(inp_var)
                                state0 = state0.unsqueeze(0)  # (1, C, H, W)
                                
                                if not use_precomputed_truth:
                                    uspec = ic.clone()

                                ic_preds = [state0[0].detach().cpu()]
                                ic_truth = [state0[0].detach().cpu()]

                                # Initialize history buffer
                                if n_history > 1:
                                    history_buffer = [state0.clone() for _ in range(n_history)]

                                # ML rollout
                                prd = state0
                                for i in range(1, _plot_steps + 1):
                                    if n_history > 1:
                                        model_input = torch.cat(history_buffer, dim=1)
                                    else:
                                        model_input = prd
                                    prd = model(model_input)
                                    if n_history > 1:
                                        history_buffer = history_buffer[1:] + [prd.clone()]
                                    ic_preds.append(prd[0].detach().cpu())

                                # Classical solver rollout or Precomputed Truth
                                if use_precomputed_truth:
                                    for i in range(1, _plot_steps + 1):
                                        ref_phys = states_chunk[i]
                                        ref = (ref_phys - inp_mean) / torch.sqrt(inp_var)
                                        ic_truth.append(ref.detach().cpu())
                                else:
                                    for i in range(1, _plot_steps + 1):
                                        uspec = dataset.solver.timestep(uspec, nsteps)
                                        ref = (dataset.solver.spec2grid(uspec) - inp_mean) / torch.sqrt(inp_var)
                                        ic_truth.append(ref.detach().cpu())

                                all_preds.append(torch.stack(ic_preds))
                                all_truth.append(torch.stack(ic_truth))

                        # Stack: (nics, steps, C, H, W)
                        preds_np = torch.stack(all_preds).numpy()
                        truth_np = torch.stack(all_truth).numpy()

                        log_dict = {}

                        # Rollout snapshots for each channel
                        for ch in range(min(3, preds_np.shape[2])):
                            fig = plot_rollout_snapshots(
                                preds_np, truth_np, channel=ch,
                                n_snapshots=6,
                            )
                            log_dict[f"rollout_{['height','vorticity','divergence'][ch]}"] = wandb.Image(fig)
                            plt.close(fig)

                        # RMSE vs lead time
                        fig = plot_rmse_vs_leadtime(
                            preds_np, truth_np,
                            channels=list(range(min(3, preds_np.shape[2]))),
                            dt_minutes=_dt_minutes
                        )
                        log_dict["rmse_vs_leadtime"] = wandb.Image(fig)
                        plt.close(fig)

                        # Variance vs lead time
                        fig = plot_variance_vs_leadtime(
                            preds_np, truth_np,
                            channels=list(range(min(3, preds_np.shape[2]))),
                            dt_minutes=_dt_minutes
                        )
                        log_dict["variance_vs_leadtime"] = wandb.Image(fig)
                        plt.close(fig)

                        if wandb.run is not None:
                            wandb.log(log_dict, commit=False)

                        plot_time = time.time() - plot_start
                        print(f"  [plot_callback] Done in {plot_time:.1f}s")

                    except Exception as e:
                        print(f"  [plot_callback] Error: {e}")
                    finally:
                        model.train()
                
                # Assign the function to the variable!
                plot_callback = generate_and_log_plots

            train_model(
                model,
                dataloader,
                loss_fn,
                metrics_fns,
                optimizer,
                gscaler,
                scheduler,
                nepochs=pretrain_epochs,
                amp_mode=amp_mode,
                log_grads=log_grads,
                logging=logging,
                device=device,
                n_history=n_history,
                noise_injection_sigma=args.noise_injection_sigma,
                noise_injection_rate=args.noise_injection_rate,
                plot_callback=plot_callback,
                scratch_dir=model_scratch_dir,
                time_diff2_lambda=args.time_diff2_lambda,
            )

            # Parse finetuning arguments into lists
            finetune_epochs_list = parse_list_arg(finetune_epochs, dtype=int)
            finetune_steps_list = parse_list_arg(finetune_steps, dtype=int)
            learning_rate_list = parse_list_arg(learning_rate, dtype=float)
            
            # Determine if we're doing multi-phase finetuning
            n_finetune_phases = max(len(finetune_epochs_list), len(finetune_steps_list))
            has_finetune = any(e > 0 for e in finetune_epochs_list) and n_finetune_phases > 0
            
            if has_finetune:
                # Broadcast single values to match number of phases
                if len(finetune_epochs_list) == 1 and n_finetune_phases > 1:
                    finetune_epochs_list = finetune_epochs_list * n_finetune_phases
                if len(finetune_steps_list) == 1 and n_finetune_phases > 1:
                    finetune_steps_list = finetune_steps_list * n_finetune_phases
                
                # Handle learning rate list:
                # - If single LR with pretrain > 0: use for pretrain, apply 0.1x for all finetune phases (backward compat)
                # - If single LR with pretrain == 0: apply 0.1x for all finetune phases (backward compat)
                # - If LR list length == n_finetune_phases: use directly for finetuning (no 0.1x)
                # - If LR list length == n_finetune_phases + 1 and pretrain > 0: first for pretrain, rest for finetuning
                if len(learning_rate_list) == 1:
                    # Backward compatibility: single LR, apply 0.1x factor for finetuning
                    finetune_lrs = [0.1 * learning_rate_list[0]] * n_finetune_phases
                elif len(learning_rate_list) == n_finetune_phases:
                    # Explicit LRs for finetuning phases only (no 0.1x)
                    finetune_lrs = learning_rate_list
                elif len(learning_rate_list) == n_finetune_phases + 1 and pretrain_epochs > 0:
                    # First LR for pretrain (already used above), rest for finetuning
                    finetune_lrs = learning_rate_list[1:]
                else:
                    raise ValueError(f"learning_rate list length ({len(learning_rate_list)}) must be 1, {n_finetune_phases}, or {n_finetune_phases + 1}")
                
                # Run each finetuning phase
                for phase_idx in range(n_finetune_phases):
                    phase_epochs = finetune_epochs_list[phase_idx]
                    phase_steps = finetune_steps_list[phase_idx]
                    phase_lr = finetune_lrs[phase_idx]
                    
                    if phase_epochs == 0:
                        continue
                    
                    nfuture = phase_steps - 1

                    if logging:
                        print(f"Finetuning phase {phase_idx + 1}/{n_finetune_phases}: {phase_epochs} epochs, {phase_steps} steps, lr={phase_lr}")

                    optimizer = torch.optim.Adam(model.parameters(), lr=phase_lr)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
                    gscaler = torch.GradScaler(enabled=(amp_mode != "none"))
                    
                    # Rebuild DataLoader so finetuning workers see the new nfuture window.
                    # Mutating dataset.nfuture in place would leave persistent workers
                    # with a stale window size, causing them to silently return
                    # single-step targets instead of trajectories.
                    if precomputed_ds is not None:
                        dataloader = _make_precomputed_loader(nfuture_val=nfuture)
                    else:
                        dataloader.dataset.nfuture = nfuture
                    

                    train_model(
                        model,
                        dataloader,
                        loss_fn,
                        metrics_fns,
                        optimizer,
                        gscaler,
                        scheduler,
                        nepochs=phase_epochs,
                        nfuture=nfuture,
                        amp_mode=amp_mode,
                        log_grads=log_grads, 
                        logging=logging,
                        device=device,
                        n_history=n_history,
                        noise_injection_sigma=args.noise_injection_sigma,
                        noise_injection_rate=args.noise_injection_rate,
                        plot_callback=plot_callback,
                        scratch_dir=model_scratch_dir,
                        time_diff2_lambda=args.time_diff2_lambda,
                    )

                    
                    # Save checkpoint after each phase
                    phase_checkpoint = os.path.join(exp_dir, f"checkpoint_phase{phase_idx + 1}_AR{phase_steps}.pt")
                    torch.save(model.state_dict(), phase_checkpoint)
                    torch.save(model.state_dict(), os.path.join(exp_dir, "checkpoint.pt"))
                    if logging:
                        print(f"Saved phase checkpoint: {phase_checkpoint}")
                    
                    # Run validation rollout after each phase (if validate_steps > 0)
                    if validate_steps > 0 and nics > 0:
                        phase_tag = f"AR{phase_steps}"
                        if logging:
                            print(f"Running validation rollout for phase {phase_idx + 1} ({phase_tag})...")
                        with torch.inference_mode():
                            losses, _, _, _ = autoregressive_inference(
                                model, dataset, loss_fn, metrics_fns, 
                                os.path.join(exp_dir, "figures"), 
                                nsteps=nsteps, autoreg_steps=validate_steps, 
                                nics=nics, device=device, 
                                run_tag=phase_tag, n_history=n_history,
                                precomputed_dataset=precomputed_ds
                            )
                            # Stability check
                            is_stable = torch.isfinite(losses).all().item() and (losses.mean() < 1e6)
                            print(f"Phase {phase_idx + 1} ({phase_tag}) Stability Check ({validate_steps} steps): {'PASSED' if is_stable else 'FAILED'}")
                
                # Reset nfuture back to 0 — rebuild loader so workers are re-forked
                # with the correct single-step window for any subsequent pretraining.
                if precomputed_ds is not None:
                    dataloader = _make_precomputed_loader(nfuture_val=0)
                else:
                    dataloader.dataset.nfuture = 0

            # Curriculum rollout training phase (optional)
            if curriculum_epochs > 0:
                if logging:
                    print(f"Curriculum rollout training {model_name}, {curriculum_epochs} epochs")
                
                # Parse curriculum schedule
                if curriculum_max_steps is not None:
                    max_steps_schedule = [int(x) for x in curriculum_max_steps.split(",")]
                else:
                    max_steps_schedule = None  # Use default linear ramp
                
                # Use 0.1x of base learning rate for curriculum (similar to old finetune behavior)
                curriculum_lr = 0.1 * parse_list_arg(learning_rate, dtype=float)[0]
                optimizer = torch.optim.Adam(model.parameters(), lr=curriculum_lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
                gscaler = torch.GradScaler("cuda", enabled=(amp_mode == "fp16"))
                
                train_curriculum_rollout(
                    model,
                    dataset,
                    loss_fn,
                    optimizer,
                    gscaler,
                    scheduler,
                    nepochs=curriculum_epochs,
                    num_examples=num_examples,
                    batch_size=batch_size,
                    amp_mode=amp_mode,
                    max_steps_schedule=max_steps_schedule,
                    logging=logging,
                    device=device,
                    n_history=n_history,
                    noise_injection_sigma=args.noise_injection_sigma,
                    noise_injection_rate=args.noise_injection_rate,
                    plot_callback=plot_callback,
                    scratch_dir=model_scratch_dir,
                )

            training_time = time.time() - start_time

            torch.save(model.state_dict(), os.path.join(exp_dir, "checkpoint.pt"))
            
            # Save a tagged checkpoint if run_tag is provided (preserves each stage)
            if run_tag:
                tagged_checkpoint = os.path.join(exp_dir, f"checkpoint_{run_tag}.pt")
                torch.save(model.state_dict(), tagged_checkpoint)
                print(f"Saved tagged checkpoint: {tagged_checkpoint}")

        # set seed
        torch.manual_seed(333)
        torch.cuda.manual_seed(333)

        # run validation
        print(f"Validating {model_name}")
        with torch.inference_mode():
            losses, metric_results, model_times, solver_times = autoregressive_inference(
                model, dataset, loss_fn, metrics_fns, os.path.join(exp_dir, "figures"), 
                nsteps=nsteps, autoreg_steps=validate_steps, nics=nics, device=device, 
                run_tag=run_tag, n_history=n_history,
                precomputed_dataset=precomputed_ds
            )

            # Stability Check
            if nics > 0:
                is_stable = torch.isfinite(losses).all().item() and (losses.mean() < 1e6) 
                print(f"Stability Check ({validate_steps} steps): {'PASSED' if is_stable else 'FAILED'}")
                metrics[model_name]["stable"] = is_stable

                # compute statistics
                metrics[model_name]["loss mean"] = torch.mean(losses).item()
                metrics[model_name]["loss std"] = torch.std(losses).item()
                if isinstance(model_times, list):
                     model_times = torch.tensor(model_times)
                if isinstance(solver_times, list):
                     solver_times = torch.tensor(solver_times)
                metrics[model_name]["model time mean"] = torch.mean(model_times).item()
                metrics[model_name]["model time std"] = torch.std(model_times).item()
                metrics[model_name]["solver time mean"] = torch.mean(solver_times).item()
                metrics[model_name]["solver time std"] = torch.std(solver_times).item()
                for metric in metric_results:
                    metrics[model_name][metric + " mean"] = torch.mean(metric_results[metric]).item()
                    metrics[model_name][metric + " std"] = torch.std(metric_results[metric]).item()

            if train:
                metrics[model_name]["training_time"] = training_time

        # Finish wandb run after validation completes to capture entire log
        if run is not None:
            run.finish()

    # output metrics to data frame
    df = pd.DataFrame(metrics)
    if not os.path.isdir(os.path.join(root_path, "output_data")):
        os.makedirs(os.path.join(root_path, "output_data"), exist_ok=True)
    df.to_pickle(os.path.join(root_path, "output_data", "metrics.pkl"))


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("forkserver", force=True)
    _log_startup("Set multiprocessing start method")
    if wandb is not None:
        _log_startup("Starting wandb.login()...")
        wandb.login()
        _log_startup("wandb.login() complete")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path", default=os.path.join(os.path.dirname(__file__), "checkpoints"), type=str, help="Override the path where checkpoints and run information are stored"
    )
    parser.add_argument("--pretrain_epochs", default=15, type=int, help="Number of pretraining epochs.")
    parser.add_argument("--finetune_epochs", default="0", type=str, help="Number of fine-tuning epochs. Can be comma-separated for multi-phase (e.g., '12,6,6').")
    parser.add_argument("--batch_size", default=4, type=int, help="Switch for overriding batch size in the configuration file.")
    parser.add_argument("--learning_rate", default="1e-4", type=str, help="Learning rate. Can be comma-separated for multi-phase (e.g., '1e-3,5e-4,2e-4').")
    parser.add_argument("--resume", action="store_true", help="Reload checkpoints.")
    parser.add_argument("--amp_mode", default="none", type=str, choices=["none", "bf16", "fp16"], help="Switch to enable AMP.")
    parser.add_argument("--validate_steps", default=96, type=int, help="Number of autoregressive steps for validation.")
    parser.add_argument("--nics", default=50, type=int, help="Number of initial conditions for validation.")
    parser.add_argument("--checkpoint_name", default="checkpoint.pt", type=str, help="Name of the checkpoint file to load.")
    parser.add_argument("--finetune_steps", default="2", type=str, help="Number of autoregressive steps for fine-tuning. Can be comma-separated for multi-phase (e.g., '2,6,12').")
    parser.add_argument("--curriculum_epochs", default=0, type=int, help="Number of curriculum rollout training epochs (0 to disable).")
    parser.add_argument("--curriculum_max_steps", default=None, type=str, help="Comma-separated schedule for max rollout steps per epoch, e.g., '1,2,4,8,16'.")
    parser.add_argument("--seed", default=None, type=int, help="Random seed for reproducibility (default: random).")
    parser.add_argument("--l2_lambda", default=1.0, type=float, help="Weight for L2 loss term (default 1.0).")
    parser.add_argument("--laplacian_lambda", default=0.0, type=float, help="Weight for Laplacian loss term (0 to disable).")
    parser.add_argument("--conservation_lambda", default=0.0, type=float, help="Weight for conservation loss term (0 to disable).")
    parser.add_argument("--ccc_lambda", default=0.0, type=float, help="Weight for CCC (Concordance Correlation Coefficient) loss term (0 to disable).")
    parser.add_argument("--spectral_lambda", default=0.0, type=float, help="Weight for Spectral loss term (0 to disable).")
    parser.add_argument("--spectral_lmin", default=0, type=int, help="Minimum wavenumber l to penalize in spectral loss.")
    parser.add_argument("--spectral_lmax", default=None, type=int, help="Maximum wavenumber l to penalize in spectral loss (default: Nyquist).")
    parser.add_argument("--spectral_decay", default=0.0, type=float, help="Exponent k for weighting spectral loss by l^k.")
    parser.add_argument("--gradient_lambda", default=0.0, type=float, help="Weight for Gradient (H1) loss term (0 to disable).")
    parser.add_argument("--vorticity_lambda", type=float, default=1.0, help="Weight for vorticity channel in L2 loss")
    parser.add_argument("--uv_lambda", type=float, default=0.0, help="Weight for UV reconstruction loss")
    parser.add_argument("--spectral_pv_lambda", type=float, default=0.0, help="Weight for Spectral PV loss")
    parser.add_argument("--mass_lambda", type=float, default=0.0, help="Weight for mass conservation loss")
    parser.add_argument("--run_tag", type=str, default=None, help="Optional run tag for logging")
    parser.add_argument("--num_examples", type=int, default=256, help="Number of training examples per epoch")
    parser.add_argument("--noise_injection_sigma", type=float, default=0.0, help="Standard deviation of Gaussian noise to inject into inputs")
    parser.add_argument("--noise_injection_rate", type=float, default=0.0, help="Probability of duplicating and corrupting samples in a batch")
    parser.add_argument("--ic_mach", type=float, default=0.2, help="Mach number for initial conditions (default: 0.2)")
    parser.add_argument("--ic_llimit", type=int, default=25, help="Maximum spectral wavenumber for IC noise (default: 25).")
    parser.add_argument("--stochastic_ic_llimit", action="store_true", help="If set, randomly sample ic_llimit for each initial condition.")
    parser.add_argument("--stochastic_ic_mach", action="store_true", help="If set, randomly sample uniform [0, 2*ic_mach] for each initial condition.")
    parser.add_argument("--wandb_run_id", default=None, type=str, help="W&B run ID to resume (8-char ID from the W&B dashboard URL).")
    parser.add_argument("--plotting_dt", default=0, type=int, help="Seconds between periodic plot logging to wandb (0 to disable).")
    parser.add_argument("--ic_spinup_max", default=0, type=int, help="Max number of solver macro-steps to advance each IC before training (0 to disable). Actual spinup per sample is randomly sampled from 0..max.")
    parser.add_argument("--time_diff2_lambda", default=0.0, type=float, help="Weight for second derivative (time) loss term.")
    parser.add_argument("--n_history", default=1, type=int, help="Number of past timesteps to use as input (default: 1).")
    parser.add_argument("--precomputed_db", type=str, default=None,
                        help="Path to pre-generated Zarr training database directory. "
                             "If set, uses precomputed data instead of on-the-fly solver generation.")
    parser.add_argument("--first_step_lambda", default=1.0, type=float, help="weight multiplier for the total loss of the very first step of a rollout")
    parser.add_argument("--first_step_only_l2_lap_uv", action="store_true", help="If enabled, ignores L2, Laplacian, and UV losses for all steps > 0.")
    parser.add_argument("--checkpoint_activation", action="store_true", help="Enable gradient checkpointing for memory savings.")
    
    args = parser.parse_args()

    # main(train=False, load_checkpoint=True, enable_amp=False, log_grads=0)
    main(
        root_path=args.root_path,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train=(int(args.pretrain_epochs) > 0 or any(int(x) > 0 for x in args.finetune_epochs.split(",")) or int(args.curriculum_epochs) > 0),
        load_checkpoint=args.resume,
        checkpoint_name=args.checkpoint_name,
        amp_mode=args.amp_mode,
        log_grads=0,
        validate_steps=args.validate_steps,
        nics=args.nics,
        finetune_steps=args.finetune_steps,
        curriculum_epochs=args.curriculum_epochs,
        curriculum_max_steps=args.curriculum_max_steps,
        seed=args.seed,
        l2_lambda=args.l2_lambda,
        laplacian_lambda=args.laplacian_lambda,
        conservation_lambda=args.conservation_lambda,
        ccc_lambda=args.ccc_lambda,
        vorticity_lambda=args.vorticity_lambda,
        uv_lambda=args.uv_lambda,
        spectral_pv_lambda=args.spectral_pv_lambda,
        mass_lambda=args.mass_lambda,
        run_tag=args.run_tag,
        num_examples=args.num_examples,
        ic_mach=args.ic_mach,
        ic_llimit=args.ic_llimit,
        stochastic_ic_llimit=args.stochastic_ic_llimit,
        stochastic_ic_mach=args.stochastic_ic_mach,
        plotting_dt=args.plotting_dt,
        ic_spinup_max=args.ic_spinup_max,
        time_diff2_lambda=args.time_diff2_lambda,
        n_history=args.n_history,
        precomputed_db_path=args.precomputed_db,
        first_step_lambda=args.first_step_lambda,
        first_step_only_l2_lap_uv=args.first_step_only_l2_lap_uv,
        checkpoint_activation=args.checkpoint_activation,
    )
