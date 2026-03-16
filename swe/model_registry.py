# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors. All rights reserved.
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

import os, sys

from functools import partial
import torch

# import baseline models (commented out - not available)
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from baseline_models import Transformer, UNet, Segformer
from torch_harmonics.examples.models import SphericalFourierNeuralOperator, LocalSphericalNeuralOperator, SphericalTransformer, SphericalUNet, SphericalSegformer
from nnorm_gam2_adapter import NNorm_GAM2_Interface
from DiscoMlpTransformer import DiscoMlpTransformer, DiscoMlpTransformerConfig
from EncodeProcessDecode import EncodeProcessDecode
from DiscoEncodeProcessDecode import DiscoEncodeProcessDecode
from HierarchicalDiscoEPD import HierarchicalDiscoEPD
from MlpTransformer import MlpTransformer, MlpTransformerConfig
from SphericalTransformerV2 import SphericalTransformer as SphericalTransformerV2

def get_baseline_models(img_size=(128, 256), in_chans=3, out_chans=3, residual_prediction=False, drop_path_rate=0., grid="equiangular"):

    # prepare dicts containing models and corresponding metrics
    model_registry = dict(
    
        disco_mlp_transformer_e64 = partial(
            DiscoMlpTransformer,
            cfg=DiscoMlpTransformerConfig(
                in_channels=in_chans,
                mlp_out_dim=31,
                disco_out_dim=31,
                in_shape=img_size,
                out_shape=img_size,
                kernel_shape=(5, 5),
                basis_type="morlet",
                basis_norm_mode="mean",
                groups=1,
                grid_in=grid,
                grid_out=grid,
                bias=True,
                disco_theta_cutoff=(5*torch.pi)/(torch.pi**0.5*img_size[0]),
                attn_theta_cutoff=(5*torch.pi)/(torch.pi**0.5*img_size[0]),
                mlp_hidden_dim=None,
                nhead=2,
                num_layers=4,
                dropout=0.0,
                add_lat_sincos=True,
                disco_gate_init=0.5,
            ),
        ),

        nnorm_gam2 = partial(
            NNorm_GAM2_Interface,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            # User Model Config:
            # M_FEATURES = 122 (Derived: in_chans + 2)
            patch_size=3,
            embed_dim=34, # 128 learnable + 2 sin/cos (divisible by 8)
            grid=grid,
            num_layers=8, # NUM_MIXER_BLOCKS
            n_heads=4,
            input_substeps=1, 
            
            # MixerEncoder Configuration
            use_mixer_encoder=False,
            num_encoder_mixer_blocks=1,
            
            # MLP Shape Configurations
            head_mlp_hidden_dims=[64, 32],
            ffn_hidden_dims=[128 * 4, 128 * 2], # [1032, 516]
            encoder_hidden_dims=[256],
            
            # Correction Layers
            num_correction_blocks=0, # Set to > 0 to enable correction layers
            correction_head_mlp_hidden_dims=[32, 16],
            correction_ffn_hidden_dims=[128],
            
            # Other defaults
            dropout_rate=0,
            grad_ckpt_level=3,
        ),
        
        # Simplified encode->process->decode model (no wrapper needed, no latent rollout)
        EPD_W9_MB1_D130_B = partial(
            EncodeProcessDecode,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            model_dim=130,           # 32 learnable + 2 sin/cos
            n_heads=8,
            num_mixer_blocks=1,
            head_mlp_hidden_dims=[48, 128, 64, 32],
            ffn_hidden_dims=[128 * 4, 128 * 2],
            encoder_hidden_dims=[256, 256],
            patch_size=9,
            dropout_rate=0,
            grad_ckpt_level=1,
            grid=grid,
            residual_prediction=residual_prediction,
        ),
        EPD_W9_MB1_D66_L = partial(
            EncodeProcessDecode,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            model_dim=66,           # 32 learnable + 2 sin/cos
            n_heads=8,
            num_mixer_blocks=1,
            head_mlp_hidden_dims=[64, 32],
            ffn_hidden_dims=[128 * 4, 128 * 2],
            encoder_hidden_dims=[128],
            patch_size=9,
            dropout_rate=0,
            grad_ckpt_level=1,
            grid=grid,
            residual_prediction=residual_prediction,
        ),
        
        # DiSCO-based encode->process->decode (geodesic spatial mixing)
        disco_epd_W9_MB1_D66_L_morlet_BTS = partial(
            DiscoEncodeProcessDecode,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            model_dim=66,           # 64 learnable + 2 sin/cos
            n_heads=8,
            num_mixer_blocks=1,
            shared_hidden=64,       # Number of DiSCO kernels
            head_mlp_hidden_dims=[32],
            ffn_hidden_dims=[128 * 4, 128 * 2],
            encoder_hidden_dims=[128],
            kernel_shape=(9, 9),
            theta_cutoff=(9*6*torch.pi)/(torch.pi**0.5*img_size[0]),
            basis_type="piecewise linear",  # Try "morlet" if seeing pole artifacts
            dropout_rate=0,
            grad_ckpt_level=0,
            grid=grid,
            residual_prediction=residual_prediction,
        ),
        # DiSCO-based encode->process->decode with FiLM latitude modulation
        disco_epd_W9_MB1_D66_L_morlet_film = partial(
            DiscoEncodeProcessDecode,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            model_dim=66,           # 64 learnable + 2 sin/cos
            n_heads=8,
            num_mixer_blocks=1,
            shared_hidden=64,       # Number of DiSCO kernels
            head_mlp_hidden_dims=[32],
            ffn_hidden_dims=[128 * 4, 128 * 2],
            encoder_hidden_dims=[128],
            kernel_shape=(9, 9),
            theta_cutoff=(9*torch.pi)/(torch.pi**0.5*img_size[0]),
            basis_type="morlet",
            dropout_rate=0,
            grad_ckpt_level=0,
            grid=grid,
            residual_prediction=residual_prediction,
            use_lat_modulation=True,  # FiLM enabled
        ),
        # DiSCO-based encode->process->decode (geodesic spatial mixing)
        disco_epd_W9_MB1_D66_L_morlet = partial(
            DiscoEncodeProcessDecode,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            model_dim=66,           # 64 learnable + 2 sin/cos
            n_heads=8,
            num_mixer_blocks=1,
            shared_hidden=64,       # Number of DiSCO kernels
            head_mlp_hidden_dims=[32],
            ffn_hidden_dims=[128 * 4, 128 * 2],
            encoder_hidden_dims=[128],
            kernel_shape=(9, 9),
            theta_cutoff=(9*torch.pi)/(torch.pi**0.5*img_size[0]),
            basis_type="morlet",  # Try "morlet" if seeing pole artifacts
            dropout_rate=0,
            grad_ckpt_level=0,
            grid=grid,
            residual_prediction=residual_prediction,
        ),
        
        # Hierarchical DiSCO: multi-scale features via stacked small-FOV DiSCO
        hierarchical_disco_epd_v1 = partial(
            HierarchicalDiscoEPD,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            model_dim=66,
            n_heads=8,
            num_mixer_blocks=1,
            num_levels=3,
            kernels_per_level=16,  # 60 total features (20+20+20)
            head_mlp_hidden_dims=[32],
            ffn_hidden_dims=[128 * 4, 128 * 2],
            encoder_hidden_dims=[128],
            kernel_shape=(5, 5),
            theta_cutoff=(5*torch.pi)/(torch.pi**0.5*img_size[0]),  # ~3x3 at equator
            basis_type="morlet",
            dropout_rate=0,
            grad_ckpt_level=3,
            grid=grid,
            residual_prediction=residual_prediction,
        ),
        sfno_sc2_layers4_e32 = partial(
            SphericalFourierNeuralOperator,
            img_size=img_size,
            grid=grid,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=2,
            embed_dim=32,
            activation_function="gelu",
            residual_prediction=residual_prediction,
            use_mlp=True,
            normalization_layer="instance_norm",
        ),
        sfno_sc3_layers4_e128 = partial(
            SphericalFourierNeuralOperator,
            img_size=img_size,
            grid=grid,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=3,
            embed_dim=128,
            activation_function="gelu",
            residual_prediction=residual_prediction,
            use_mlp=True,
            normalization_layer="instance_norm",
        ),
        sfno_sc3_layers4_e256 = partial(
            SphericalFourierNeuralOperator,
            img_size=img_size,
            grid=grid,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=3,
            embed_dim=256,
            activation_function="gelu",
            residual_prediction=residual_prediction,
            use_mlp=True,
            normalization_layer="instance_norm",
        ),
        lsno_sc2_layers4_e32 = partial(
            LocalSphericalNeuralOperator,
            img_size=img_size,
            grid=grid,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=2,
            embed_dim=32,
            activation_function="gelu",
            residual_prediction=residual_prediction,
            use_mlp=True,
            normalization_layer="instance_norm",
            kernel_shape=(5, 4),
            encoder_kernel_shape=(5, 4),
            filter_basis_type="piecewise linear",
            upsample_sht=False,
        ),
        s2unet_sc2_layers4_e128 = partial(
            SphericalUNet,
            img_size=img_size,
            grid=grid,
            grid_internal="equiangular",
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dims=[16, 32, 64, 128],
            depths=[2, 2, 2, 2],
            scale_factor=2,
            activation_function="gelu",
            kernel_shape=(5, 4),
            filter_basis_type="piecewise linear",
            drop_path_rate=0.1,
            drop_conv_rate=0.2,
            drop_dense_rate=0.5,
            transform_skip=False,
            upsampling_mode="conv",
            downsampling_mode="conv",
        ),
        
        s2transformer_sc2_layers4_e128 = partial(
            SphericalTransformer,
            img_size=img_size,
            grid=grid,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=2,
            embed_dim=128,
            activation_function="gelu",
            residual_prediction=residual_prediction,
            pos_embed="spectral",
            use_mlp=True,
            normalization_layer="instance_norm",
            encoder_kernel_shape=(5, 4),
            filter_basis_type="piecewise linear",
            drop_path_rate=drop_path_rate,
            upsample_sht=False,
            attention_mode="global",
            bias=False
        ),
        s2transformer_sc2_layers4_e256 = partial(
            SphericalTransformer,
            img_size=img_size,
            grid=grid,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=2,
            embed_dim=256,
            activation_function="gelu",
            residual_prediction=residual_prediction,
            pos_embed="spectral",
            use_mlp=True,
            normalization_layer="instance_norm",
            encoder_kernel_shape=(5, 4),
            filter_basis_type="piecewise linear",
            drop_path_rate=drop_path_rate,
            upsample_sht=False,
            attention_mode="global",
            bias=False
        ),
        
        s2ntransformer_sc2_layers4_e128 = partial(
            SphericalTransformer,
            img_size=img_size,
            grid=grid,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=1,
            embed_dim=128,
            activation_function="gelu",
            residual_prediction=residual_prediction,
            pos_embed="spectral",
            use_mlp=True,
            normalization_layer="instance_norm",
            encoder_kernel_shape=(5, 5),
            filter_basis_type="morlet",
            drop_path_rate=drop_path_rate,
            upsample_sht=False,
            attention_mode="neighborhood",
            bias=False,
            theta_cutoff=(4*torch.pi)/(torch.pi**0.5*img_size[0]),
        ),

        s2ntransformer_sc2_layers4_e128_noresid = partial(
            SphericalTransformer,
            img_size=img_size,
            grid=grid,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=0.5,
            embed_dim=128,
            activation_function="gelu",
            residual_prediction=False,
            pos_embed="spectral",
            use_mlp=True,
            normalization_layer="instance_norm",
            encoder_kernel_shape=(5, 5),
            filter_basis_type="morlet",
            drop_path_rate=drop_path_rate,
            upsample_sht=False,
            attention_mode="neighborhood",
            bias=False,
            theta_cutoff=(4*torch.pi)/(torch.pi**0.5*img_size[0]),
        ),

        s2ntransformer_sc1_layers4_e128_global_ctrl = partial(
            SphericalTransformerV2,
            img_size=img_size,
            grid=grid,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=1,
            embed_dim=128,
            activation_function="gelu",
            residual_prediction=residual_prediction,
            pos_embed="spectral",
            use_mlp=True,
            normalization_layer="instance_norm",
            encoder_kernel_shape=(5, 5),
            filter_basis_type="morlet",
            drop_path_rate=drop_path_rate,
            upsample_sht=False,
            attention_mode="global",
            bias=False,
            theta_cutoff=(4*torch.pi)/(torch.pi**0.5*img_size[0]),
        ),

        s2ntransformer_sc2_layers4_e128_k55_tc9 = partial(
            SphericalTransformer,
            img_size=img_size,
            grid=grid,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=1,
            embed_dim=128,
            activation_function="gelu",
            residual_prediction=residual_prediction,
            pos_embed="spectral",
            use_mlp=True,
            normalization_layer="instance_norm",
            encoder_kernel_shape=(5, 5),
            filter_basis_type="morlet",
            drop_path_rate=drop_path_rate,
            upsample_sht=False,
            attention_mode="neighborhood",
            bias=False,
            theta_cutoff=(9*torch.pi)/(torch.pi**0.5*img_size[0]),
        ),

        s2ntransformer_sc2_layers4_e128_k97_tc4 = partial(
            SphericalTransformer,
            img_size=img_size,
            grid=grid,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=1,
            embed_dim=128,
            activation_function="gelu",
            residual_prediction=residual_prediction,
            pos_embed="spectral",
            use_mlp=True,
            normalization_layer="instance_norm",
            encoder_kernel_shape=(9, 7),
            filter_basis_type="morlet",
            drop_path_rate=drop_path_rate,
            upsample_sht=False,
            attention_mode="neighborhood",
            bias=False,
            theta_cutoff=(4*torch.pi)/(torch.pi**0.5*img_size[0]),
        ),

        s2ntransformer_sc2_layers4_e128_scale2 = partial(
            SphericalTransformer,
            img_size=img_size,
            grid=grid,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=2,
            embed_dim=128,
            activation_function="gelu",
            residual_prediction=residual_prediction,
            pos_embed="spectral",
            use_mlp=True,
            normalization_layer="instance_norm",
            encoder_kernel_shape=(5, 5),
            filter_basis_type="morlet",
            drop_path_rate=drop_path_rate,
            upsample_sht=False,
            attention_mode="neighborhood",
            bias=False,
            theta_cutoff=(4*torch.pi)/(torch.pi**0.5*img_size[0]),
        ),

        s2ntransformer_sc2_layers4_e256 = partial(
            SphericalTransformer,
            img_size=img_size,
            grid=grid,
            in_chans=in_chans,
            out_chans=out_chans,
            num_layers=4,
            scale_factor=2,
            embed_dim=256,
            activation_function="gelu",
            residual_prediction=True,
            pos_embed="spectral",
            use_mlp=True,
            normalization_layer="instance_norm",
            encoder_kernel_shape=(5, 4),
            filter_basis_type="piecewise linear",
            drop_path_rate=drop_path_rate,
            upsample_sht=False,
            attention_mode="neighborhood",
            bias=False
        ),
        
        # NOTE: The following models require baseline_models.py (not available)
        # Commented out: transformer_*, ntransformer_*, segformer_*, nsegformer_*, vit_*
        
        s2segformer_sc2_layers4_e128 = partial(
            SphericalSegformer,
            img_size=img_size,
            grid=grid,
            grid_internal="equiangular",
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dims=[16, 32, 64, 128],
            heads=[1, 2, 4, 8],
            depths=[3, 4, 6, 3],
            scale_factor=2,
            activation_function="gelu",
            kernel_shape=(5, 4),
            filter_basis_type="piecewise linear",
            mlp_ratio=4.0,
            att_drop_rate=0.0,
            drop_path_rate=0.1,
            attention_mode="global",
            bias=False
        ),
        s2segformer_sc2_layers4_e256 = partial(
            SphericalSegformer,
            img_size=img_size,
            grid=grid,
            grid_internal="equiangular",
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dims=[32, 64, 128, 256],
            heads=[1, 2, 4, 8],
            depths=[3, 4, 6, 3],
            scale_factor=2,
            activation_function="gelu",
            kernel_shape=(5, 4),
            filter_basis_type="piecewise linear",
            mlp_ratio=4.0,
            att_drop_rate=0.0,
            drop_path_rate=0.1,
            attention_mode="global",
            bias=False
        ),
        
        s2nsegformer_sc2_layers4_e128 = partial(
            SphericalSegformer,
            img_size=img_size,
            grid=grid,
            grid_internal="equiangular",
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dims=[16, 32, 64, 128],
            heads=[1, 2, 4, 8],
            depths=[3, 4, 6, 3],
            scale_factor=2,
            activation_function="gelu",
            kernel_shape=(5, 4),
            filter_basis_type="piecewise linear",
            mlp_ratio=4.0,
            att_drop_rate=0.0,
            drop_path_rate=0.1,
            attention_mode="neighborhood",
            bias=False
        ),
        s2nsegformer_sc2_layers4_e256 = partial(
            SphericalSegformer,
            img_size=img_size,
            grid=grid,
            grid_internal="equiangular",
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dims=[32, 64, 128, 256],
            heads=[1, 2, 4, 8],
            depths=[3, 4, 6, 3],
            scale_factor=2,
            activation_function="gelu",
            kernel_shape=(5, 4),
            filter_basis_type="piecewise linear",
            mlp_ratio=4.0,
            att_drop_rate=0.0,
            drop_path_rate=0.1,
            attention_mode="neighborhood",
            bias=False
        ),
############ DISCO EPD  testiin ############
        # DiSCO-based encode->process->decode with FiLM latitude modulation
        film = partial(
            DiscoEncodeProcessDecode,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            model_dim=66,           # 64 learnable + 2 sin/cos
            n_heads=8,
            num_mixer_blocks=1,
            shared_hidden=64,       # Number of DiSCO kernels
            head_mlp_hidden_dims=[32],
            ffn_hidden_dims=[128 * 4, 128 * 2],
            encoder_hidden_dims=[128],
            kernel_shape=(9, 9),
            theta_cutoff=(9*torch.pi)/(torch.pi**0.5*img_size[0]),
            basis_type="morlet",
            dropout_rate=0,
            grad_ckpt_level=5,
            grid=grid,
            residual_prediction=residual_prediction,
            use_lat_modulation=True,  # FiLM enabled
            use_spectral_filter=False,
            spectral_lmax=img_size[0]//3,
            use_hyperdiffusion=False,
            hyperdiff_dt=600.0,  # Match ML prediction step (dt in TH_SWE.py)
            hyperdiff_order=4,
            enforce_conservation=False,
        ),
        control = partial(
            DiscoEncodeProcessDecode,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            model_dim=66,           # 64 learnable + 2 sin/cos
            n_heads=8,
            num_mixer_blocks=1,
            shared_hidden=64,       # Number of DiSCO kernels
            head_mlp_hidden_dims=[32],
            ffn_hidden_dims=[128 * 4, 128 * 2],
            encoder_hidden_dims=[128],
            kernel_shape=(9, 9),
            theta_cutoff=(9*torch.pi)/(torch.pi**0.5*img_size[0]),
            basis_type="morlet",
            dropout_rate=0,
            grad_ckpt_level=0,
            grid=grid,
            residual_prediction=residual_prediction,
            use_lat_modulation=False,  # FiLM enabled
            use_spectral_filter=False,
            spectral_lmax=img_size[0]//3
        ),
        filtered = partial(
            DiscoEncodeProcessDecode,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            model_dim=66,           # 64 learnable + 2 sin/cos
            n_heads=8,
            num_mixer_blocks=1,
            shared_hidden=64,       # Number of DiSCO kernels
            head_mlp_hidden_dims=[32],
            ffn_hidden_dims=[128 * 4, 128 * 2],
            encoder_hidden_dims=[128],
            kernel_shape=(9, 9),
            theta_cutoff=(9*torch.pi)/(torch.pi**0.5*img_size[0]),
            basis_type="morlet",
            dropout_rate=0,
            grad_ckpt_level=0,
            grid=grid,
            residual_prediction=residual_prediction,
            use_lat_modulation=False,  # FiLM enabled
            use_spectral_filter=True,
            spectral_lmax=img_size[0]//3
        ),
        film_filtered = partial(
            DiscoEncodeProcessDecode,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            model_dim=66,           # 64 learnable + 2 sin/cos
            n_heads=8,
            num_mixer_blocks=1,
            shared_hidden=64,       # Number of DiSCO kernels
            head_mlp_hidden_dims=[32],
            ffn_hidden_dims=[128 * 4, 128 * 2],
            encoder_hidden_dims=[128],
            kernel_shape=(9, 9),
            theta_cutoff=(9*torch.pi)/(torch.pi**0.5*img_size[0]),
            basis_type="morlet",
            dropout_rate=0,
            grad_ckpt_level=5,
            grid=grid,
            residual_prediction=residual_prediction,
            use_lat_modulation=True,  # FiLM enabled
            use_spectral_filter=True,
            spectral_lmax=img_size[0]//3,
            use_hyperdiffusion=False,
            hyperdiff_dt=600.0,  # Match ML prediction step (dt in TH_SWE.py)
            hyperdiff_order=4,
        ),
        film_filtered_diffusion = partial(
            DiscoEncodeProcessDecode,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            model_dim=66,           # 64 learnable + 2 sin/cos
            n_heads=8,
            num_mixer_blocks=1,
            shared_hidden=64,       # Number of DiSCO kernels
            head_mlp_hidden_dims=[32],
            ffn_hidden_dims=[128 * 4, 128 * 2],
            encoder_hidden_dims=[128],
            kernel_shape=(9, 9),
            theta_cutoff=(9*torch.pi)/(torch.pi**0.5*img_size[0]),
            basis_type="morlet",
            dropout_rate=0,
            grad_ckpt_level=2,
            grid=grid,
            residual_prediction=residual_prediction,
            use_lat_modulation=True,  # FiLM enabled
            use_spectral_filter=True,
            spectral_lmax=img_size[0]//3,
            use_hyperdiffusion=True,
            hyperdiff_dt=600.0,  # Match ML prediction step (dt in TH_SWE.py)
            hyperdiff_order=4,
        ),
        film_filtered_BHMLP = partial(
            DiscoEncodeProcessDecode,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            model_dim=66,           # 64 learnable + 2 sin/cos
            n_heads=8,
            num_mixer_blocks=1,
            shared_hidden=64,       # Number of DiSCO kernels
            head_mlp_hidden_dims=[128,64,32],
            ffn_hidden_dims=[128 * 4, 128 * 2],
            encoder_hidden_dims=[128],
            kernel_shape=(9, 9),
            theta_cutoff=(9*torch.pi)/(torch.pi**0.5*img_size[0]),
            basis_type="morlet",
            dropout_rate=0,
            grad_ckpt_level=0,
            grid=grid,
            residual_prediction=residual_prediction,
            use_lat_modulation=True,  # FiLM enabled
            use_spectral_filter=True,
            spectral_lmax=img_size[0]//3
        ),
        film_filtered_W11_K11 = partial(
            DiscoEncodeProcessDecode,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            model_dim=66,           # 64 learnable + 2 sin/cos
            n_heads=8,
            num_mixer_blocks=1,
            shared_hidden=64,       # Number of DiSCO kernels
            head_mlp_hidden_dims=[32],
            ffn_hidden_dims=[128 * 4, 128 * 2],
            encoder_hidden_dims=[128],
            kernel_shape=(11, 11),
            theta_cutoff=(11*torch.pi)/(torch.pi**0.5*img_size[0]),
            basis_type="morlet",
            dropout_rate=0,
            grad_ckpt_level=0,
            grid=grid,
            residual_prediction=residual_prediction,
            use_lat_modulation=True,  # FiLM enabled
            use_spectral_filter=True,
            spectral_lmax=img_size[0]//3
        ),
        film_filtered_W7_K7 = partial(
            DiscoEncodeProcessDecode,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            model_dim=66,           # 64 learnable + 2 sin/cos
            n_heads=8,
            num_mixer_blocks=1,
            shared_hidden=64,       # Number of DiSCO kernels
            head_mlp_hidden_dims=[32],
            ffn_hidden_dims=[128 * 4, 128 * 2],
            encoder_hidden_dims=[128],
            kernel_shape=(7, 7),
            theta_cutoff=(7*torch.pi)/(torch.pi**0.5*img_size[0]),
            basis_type="morlet",
            dropout_rate=0,
            grad_ckpt_level=0,
            grid=grid,
            residual_prediction=residual_prediction,
            use_lat_modulation=True,  # FiLM enabled
            use_spectral_filter=True,
            spectral_lmax=img_size[0]//3
        ),
        film_filtered_5x5 = partial(
            DiscoEncodeProcessDecode,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            model_dim=66,           # 64 learnable + 2 sin/cos
            n_heads=8,
            num_mixer_blocks=1,
            shared_hidden=64,       # Number of DiSCO kernels
            head_mlp_hidden_dims=[32],
            ffn_hidden_dims=[128 * 4, 128 * 2],
            encoder_hidden_dims=[128],
            kernel_shape=(5, 5),
            theta_cutoff=(9*torch.pi)/(torch.pi**0.5*img_size[0]),
            basis_type="morlet",
            dropout_rate=0,
            grad_ckpt_level=0,
            grid=grid,
            residual_prediction=residual_prediction,
            use_lat_modulation=True,  # FiLM enabled
            use_spectral_filter=True,
            spectral_lmax=img_size[0]//3
        ),
        film_filtered_e34 = partial(
            DiscoEncodeProcessDecode,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            model_dim=34,           # 64 learnable + 2 sin/cos
            n_heads=4,
            num_mixer_blocks=1,
            shared_hidden=64,       # Number of DiSCO kernels
            head_mlp_hidden_dims=[32],
            ffn_hidden_dims=[128 * 4, 128 * 2],
            encoder_hidden_dims=[128],
            kernel_shape=(9, 9),
            theta_cutoff=(9*torch.pi)/(torch.pi**0.5*img_size[0]),
            basis_type="morlet",
            dropout_rate=0,
            grad_ckpt_level=0,
            grid=grid,
            residual_prediction=residual_prediction,
            use_lat_modulation=True,  # FiLM enabled
            use_spectral_filter=True,
            spectral_lmax=img_size[0]//3
        ),
        film_filtered_e34_4L = partial(
            DiscoEncodeProcessDecode,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            model_dim=34,           # 64 learnable + 2 sin/cos
            n_heads=4,
            num_mixer_blocks=4,
            shared_hidden=64,       # Number of DiSCO kernels
            head_mlp_hidden_dims=[32],
            ffn_hidden_dims=[128 * 4, 128 * 2],
            encoder_hidden_dims=[128],
            kernel_shape=(9, 9),
            theta_cutoff=(9*torch.pi)/(torch.pi**0.5*img_size[0]),
            basis_type="morlet",
            dropout_rate=0,
            grad_ckpt_level=0,
            grid=grid,
            residual_prediction=residual_prediction,
            use_lat_modulation=True,  # FiLM enabled
            use_spectral_filter=True,
            spectral_lmax=img_size[0]//3
        ),
        film_filtered_2L = partial(
            DiscoEncodeProcessDecode,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            model_dim=66,           # 64 learnable + 2 sin/cos
            n_heads=8,
            num_mixer_blocks=2,
            shared_hidden=64,       # Number of DiSCO kernels
            head_mlp_hidden_dims=[32],
            ffn_hidden_dims=[128 * 4, 128 * 2],
            encoder_hidden_dims=[128],
            kernel_shape=(9, 9),
            theta_cutoff=(9*torch.pi)/(torch.pi**0.5*img_size[0]),
            basis_type="morlet",
            dropout_rate=0,
            grad_ckpt_level=0,
            grid=grid,
            residual_prediction=residual_prediction,
            use_lat_modulation=True,  # FiLM enabled
            use_spectral_filter=True,
            spectral_lmax=img_size[0]//3
        ),
        film_filtered_2L_e126 = partial(
            DiscoEncodeProcessDecode,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            model_dim=130,           # 64 learnable + 2 sin/cos
            n_heads=8,
            num_mixer_blocks=2,
            shared_hidden=64,       # Number of DiSCO kernels
            head_mlp_hidden_dims=[32],
            ffn_hidden_dims=[128 * 4, 128 * 2],
            encoder_hidden_dims=[128],
            kernel_shape=(9, 9),
            theta_cutoff=(9*torch.pi)/(torch.pi**0.5*img_size[0]),
            basis_type="morlet",
            dropout_rate=0,
            grad_ckpt_level=0,
            grid=grid,
            residual_prediction=residual_prediction,
            use_lat_modulation=True,  # FiLM enabled
            use_spectral_filter=True,
            spectral_lmax=img_size[0]//3
        ),
        zernike = partial(
            DiscoEncodeProcessDecode,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            model_dim=66,           # 64 learnable + 2 sin/cos
            n_heads=8,
            num_mixer_blocks=1,
            shared_hidden=64,       # Number of DiSCO kernels
            head_mlp_hidden_dims=[32],
            ffn_hidden_dims=[128 * 4, 128 * 2],
            encoder_hidden_dims=[128],
            kernel_shape=12,
            theta_cutoff=(9*torch.pi)/(torch.pi**0.5*img_size[0]),
            basis_type="zernike",
            dropout_rate=0,
            grad_ckpt_level=0,
            grid=grid,
            residual_prediction=residual_prediction,
            use_lat_modulation=False,  # FiLM enabled
            use_spectral_filter=False,
            spectral_lmax=img_size[0]//3
        ),
        film_filtered_zernike = partial(
            DiscoEncodeProcessDecode,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            model_dim=66,           # 64 learnable + 2 sin/cos
            n_heads=8,
            num_mixer_blocks=1,
            shared_hidden=64,       # Number of DiSCO kernels
            head_mlp_hidden_dims=[32],
            ffn_hidden_dims=[128 * 4, 128 * 2],
            encoder_hidden_dims=[128],
            kernel_shape=12,
            theta_cutoff=(9*torch.pi)/(torch.pi**0.5*img_size[0]),
            basis_type="zernike",
            dropout_rate=0,
            grad_ckpt_level=0,
            grid=grid,
            residual_prediction=residual_prediction,
            use_lat_modulation=True,  # FiLM enabled
            use_spectral_filter=True,
            spectral_lmax=img_size[0]//3
        ),
        disco_mlp_transformer = partial(
            DiscoMlpTransformer,
            cfg=DiscoMlpTransformerConfig(
                in_channels=in_chans,
                mlp_out_dim=31,
                disco_out_dim=31,
                in_shape=img_size,
                out_shape=img_size,
                kernel_shape=(4, 4),
                basis_type="morlet",
                basis_norm_mode="mean",
                groups=1,
                grid_in=grid,
                grid_out=grid,
                bias=True,
                disco_theta_cutoff=(4*torch.pi)/(torch.pi**0.5*img_size[0]),
                attn_theta_cutoff=(4*torch.pi)/(torch.pi**0.5*img_size[0]),
                mlp_hidden_dim=128,
                nhead=2,
                num_layers=3,
                dropout=0.0,
                add_lat_sincos=True,
                disco_gate_init=0.3,
            ),
        ),
        disco_mlp_transformer_e128 = partial(
            DiscoMlpTransformer,
            cfg=DiscoMlpTransformerConfig(
                in_channels=in_chans,
                mlp_out_dim=63,
                disco_out_dim=63,
                in_shape=img_size,
                out_shape=img_size,
                kernel_shape=(4, 4),
                basis_type="morlet",
                basis_norm_mode="mean",
                groups=1,
                grid_in=grid,
                grid_out=grid,
                bias=True,
                disco_theta_cutoff=(4*torch.pi)/(torch.pi**0.5*img_size[0]),
                attn_theta_cutoff=(4*torch.pi)/(torch.pi**0.5*img_size[0]),
                mlp_hidden_dim=128,
                nhead=2,
                num_layers=3,
                dropout=0.0,
                add_lat_sincos=True,
                disco_gate_init=0.3,
            ),
        ),
        # Hierarchical DiSCO: multi-scale features via stacked small-FOV DiSCO
        hierarchical_disco_epd = partial(
            HierarchicalDiscoEPD,
            img_size=img_size,
            in_chans=in_chans,
            out_chans=out_chans,
            model_dim=66,
            n_heads=8,
            num_mixer_blocks=1,
            num_levels=2,
            kernels_per_level=[32, 16],  # 60 total features (20+20+20)
            head_mlp_hidden_dims=[32],
            ffn_hidden_dims=[128 * 4, 128 * 2],
            encoder_hidden_dims=[128],
            kernel_shape=[(9, 9), (4,4)],
            theta_cutoff=[(4*torch.pi)/(torch.pi**0.5*img_size[0]),(4*torch.pi)/(torch.pi**0.5*img_size[0])],  # ~3x3 at equator
            basis_type="morlet",
            dropout_rate=0,
            grad_ckpt_level=3,
            grid=grid,
            residual_prediction=residual_prediction,
            use_lat_modulation=True,  # FiLM enabled
            use_spectral_filter=True,
            spectral_lmax=img_size[0]//3
        ),
        
        # ============ MULTI-HISTORY MODELS ============
        # These models accept 3 stacked timesteps as input (n_history=3)
        # Input shape: (B, 9, H, W) where 9 = 3 timesteps × 3 channels
        # Output shape: (B, 3, H, W) - single next step prediction
        # 
        # The model name suffix "_history3" indicates n_history=3 requirement.
        # Autoregressive inference must maintain a 3-step sliding window buffer.
        
        film_history3 = partial(
            DiscoEncodeProcessDecode,
            img_size=img_size,
            in_chans=9,  # 3 timesteps × 3 channels
            out_chans=out_chans,  # Still predict single timestep
            model_dim=66,
            n_heads=8,
            num_mixer_blocks=1,
            shared_hidden=64,
            head_mlp_hidden_dims=[32],
            ffn_hidden_dims=[128 * 4, 128 * 2],
            encoder_hidden_dims=[128],
            kernel_shape=(9, 9),
            theta_cutoff=(9*torch.pi)/(torch.pi**0.5*img_size[0]),
            basis_type="morlet",
            dropout_rate=0,
            grad_ckpt_level=5,
            grid=grid,
            residual_prediction=residual_prediction,
            use_lat_modulation=True,
            use_spectral_filter=False,
            spectral_lmax=img_size[0]//3,
            use_hyperdiffusion=False,
            n_history=3,  # Model expects 3 stacked timesteps
        ),

        film_history3_conservation = partial(
            DiscoEncodeProcessDecode,
            img_size=img_size,
            in_chans=9,  # 3 timesteps × 3 channels
            out_chans=out_chans,  # Still predict single timestep
            model_dim=66,
            n_heads=8,
            num_mixer_blocks=1,
            shared_hidden=64,
            head_mlp_hidden_dims=[32],
            ffn_hidden_dims=[128 * 4, 128 * 2],
            encoder_hidden_dims=[128],
            kernel_shape=(9, 9),
            theta_cutoff=(9*torch.pi)/(torch.pi**0.5*img_size[0]),
            basis_type="morlet",
            dropout_rate=0,
            grad_ckpt_level=5,
            grid=grid,
            residual_prediction=residual_prediction,
            use_lat_modulation=True,
            use_spectral_filter=False,
            spectral_lmax=img_size[0]//3,
            use_hyperdiffusion=False,
            n_history=3,  # Model expects 3 stacked timesteps
            enforce_conservation=True, 
        ),

        film_conservation = partial(
            DiscoEncodeProcessDecode,
            img_size=img_size,
            in_chans=in_chans,  # 3 channels for single timestep
            out_chans=out_chans,
            model_dim=66,
            n_heads=8,
            num_mixer_blocks=1,
            shared_hidden=64,
            head_mlp_hidden_dims=[32],
            ffn_hidden_dims=[128 * 4, 128 * 2],
            encoder_hidden_dims=[128],
            kernel_shape=(9, 9),
            theta_cutoff=(9*torch.pi)/(torch.pi**0.5*img_size[0]),
            basis_type="morlet",
            dropout_rate=0,
            grad_ckpt_level=5,
            grid=grid,
            residual_prediction=residual_prediction,
            use_lat_modulation=True,
            use_spectral_filter=False,
            spectral_lmax=img_size[0]//3,
            use_hyperdiffusion=False,
            n_history=1,  # Single timestep
            enforce_conservation=True, 
        ),
        film_conservation_spectral_norm = partial(
            DiscoEncodeProcessDecode,
            img_size=img_size,
            in_chans=in_chans,  # 3 channels for single timestep
            out_chans=out_chans,
            model_dim=66,
            n_heads=8,
            num_mixer_blocks=1,
            shared_hidden=64,
            head_mlp_hidden_dims=[32],
            ffn_hidden_dims=[128 * 4, 128 * 2],
            encoder_hidden_dims=[128],
            kernel_shape=(9, 9),
            theta_cutoff=(9*torch.pi)/(torch.pi**0.5*img_size[0]),
            basis_type="morlet",
            dropout_rate=0,
            grad_ckpt_level=5,
            grid=grid,
            residual_prediction=residual_prediction,
            use_lat_modulation=True,
            use_spectral_filter=False,
            spectral_lmax=img_size[0]//3,
            use_hyperdiffusion=False,
            n_history=1,  # Single timestep
            enforce_conservation=True, 
            use_spectral_norm=True,
        ),

        mlp_transformer_e64_h2 = partial(
            MlpTransformer,
            cfg=MlpTransformerConfig(
                in_channels=in_chans,
                model_dim=64,
                in_shape=img_size,
                out_shape=img_size,
                # Shared MLP: in_chans -> 512 -> 128 (model_dim)
                mlp_hidden_dim=None,
                nhead=2,
                num_layers=2,
                dropout=0.0,
                add_lat_sincos=True,
                attn_theta_cutoff=(9*torch.pi)/(torch.pi**0.5*img_size[0]),
            ),
        ),
        mlp_transformer_e126_h4 = partial(
            MlpTransformer,
            cfg=MlpTransformerConfig(
                in_channels=in_chans,
                model_dim=126,
                in_shape=img_size,
                out_shape=img_size,
                # Shared MLP: in_chans -> 512 -> 128 (model_dim)
                mlp_hidden_dim=None,
                nhead=4,
                num_layers=2,
                dropout=0.0,
                add_lat_sincos=True,
                attn_theta_cutoff=(4*torch.pi)/(torch.pi**0.5*img_size[0]),
            ),
        ),
        
        mlp_transformer_e126_h4_no_lat = partial(
            MlpTransformer,
            cfg=MlpTransformerConfig(
                in_channels=in_chans,
                model_dim=128,
                in_shape=img_size,
                out_shape=img_size,
                # Shared MLP: in_chans -> 512 -> 128 (model_dim)
                mlp_hidden_dim=None,
                nhead=4,
                num_layers=2,
                dropout=0.0,
                add_lat_sincos=False,
                attn_theta_cutoff=(4*torch.pi)/(torch.pi**0.5*img_size[0]),
            ),
        ),
        mlp_transformer_e126_h4_4L = partial(
            MlpTransformer,
            cfg=MlpTransformerConfig(
                in_channels=in_chans,
                model_dim=126,
                in_shape=img_size,
                out_shape=img_size,
                # Shared MLP: in_chans -> 512 -> 128 (model_dim)
                mlp_hidden_dim=None,
                nhead=4,
                num_layers=4,
                dropout=0.0,
                add_lat_sincos=False,
                attn_theta_cutoff=(4*torch.pi)/(torch.pi**0.5*img_size[0]),
            ),
        ),
        mlp_transformer_e126_h1_4L_1H = partial(
            MlpTransformer,
            cfg=MlpTransformerConfig(
                in_channels=in_chans,
                model_dim=126,
                in_shape=img_size,
                out_shape=img_size,
                # Shared MLP: in_chans -> 512 -> 128 (model_dim)
                mlp_hidden_dim=None,
                nhead=1,
                num_layers=4,
                dropout=0.0,
                add_lat_sincos=False,
                attn_theta_cutoff=(4*torch.pi)/(torch.pi**0.5*img_size[0]),
            ),
        ),
        


        s2unet_local_physics_small = partial(
            SphericalUNet,
            img_size=img_size,
            grid=grid,
            grid_internal="equiangular",
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dims=[64],       # Single level, constant width
            depths=[4],             # 4 blocks deep for non-linearity
            scale_factor=1,         # No resolution change
            activation_function="gelu",
            kernel_shape=(3, 3),    # Small local kernel for restricted FOV
            filter_basis_type="morlet",
            drop_path_rate=0.0,     # No drop path for now
            drop_conv_rate=0.0,
            drop_dense_rate=0.0,
            transform_skip=False,   # No skip connection transform needed
            upsampling_mode="conv", # Learnable "upsampling" (identity at scale 1)
            downsampling_mode="conv", # Learnable "downsampling" (identity at scale 1)
        ),
    )

    return model_registry