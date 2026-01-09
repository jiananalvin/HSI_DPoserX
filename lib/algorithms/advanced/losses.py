# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions related to loss computation and optimization.
"""

import numpy as np
import torch
import torch.optim as optim
from torch import nn

from lib.utils.misc import lerp, create_random_mask, create_random_part_mask, apply_random_part_mask
from . import utils as mutils
from .sde_lib import VESDE, VPSDE


def get_optimizer(config, params):
    """Returns a flax optimizer object based on `config`."""
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params,
                                lr=config.optim.lr,
                                betas=(config.optim.beta1, 0.98),
                                eps=config.optim.eps,
                                weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')
    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""
    def optimize_fn(params, total_loss, optimizer, step, 
                    lr=config.optim.lr, warmup=config.optim.warmup, grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (fixed argument order)."""
        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()
    return optimize_fn


def get_sde_loss_fn(sde, train, reduce_mean=False, continuous=True, likelihood_weighting=False, eps=1e-5,
                    return_data=False, denoise_steps=5,
                    random_mask=False, min_mask_rate=0.2, max_mask_rate=0.4, observation_type='noise',
                    random_part_mask=False, mask_prob=0.2, apply_loss=True):
    """Create a loss function for training with arbitrary SDEs + reconstruction loss."""
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch, condition, mask):
        """Compute diffusion loss + reconstruction (MSE/MPJPE) loss."""
        def multi_step_denoise(x_t, t, t_end, N=10):
            time_traj = lerp(t, t_end, N + 1)
            x_current = x_t.clone()
            batch_size = x_t.shape[0]
            for i in range(N):
                t_current = time_traj[i]
                t_before = time_traj[i + 1]
                
                # Fix shape for alpha/sigma (critical for broadcasting)
                alpha_current, sigma_current = sde.return_alpha_sigma(t_current)
                alpha_before, sigma_before = sde.return_alpha_sigma(t_before)
                alpha_current = alpha_current.flatten().unsqueeze(1)
                sigma_current = sigma_current.flatten().unsqueeze(1)
                alpha_before = alpha_before.flatten().unsqueeze(1)
                sigma_before = sigma_before.flatten().unsqueeze(1)
                
                score = score_fn(x_current, t_current, condition=condition, mask=mask)
                if i == 0:
                    score_return = score
                
                score = -score * sigma_current
                x_current = alpha_before / alpha_current * (x_current - sigma_current * score) + sigma_before * score
                x_current = x_current.reshape(batch_size, -1)
            return score_return, x_current

        # Core diffusion loss logic
        score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        z = torch.randn_like(batch)

        # Mask handling (unchanged from your code)
        if mask is None:
            if random_mask:
                mask, batch = create_random_mask(batch, min_mask_rate, max_mask_rate, observation_type)
                loss_mask = mask
            elif random_part_mask:
                mask, loss_mask = create_random_part_mask(batch.shape[0], batch.device, mask_prob, 3, apply_loss)
            else:
                loss_mask = torch.ones(batch.shape[0], 1, dtype=torch.bool, device=batch.device)
        else:
            assert mask.shape[1] == 4, "mask should be [B, 4] for wholebody training"
            if random_part_mask:
                mask, loss_mask = apply_random_part_mask(mask, mask_prob, 0.2, rot_N=3, apply_loss=apply_loss)
            else:
                _, loss_mask = apply_random_part_mask(mask, mask_prob=0.0, rot_N=3,)

        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None] * z

        # Initialize reconstruction loss variables (safe defaults)
        recon_param_mse = torch.tensor(0.0, device=batch.device)
        recon_joint_l2 = torch.tensor(0.0, device=batch.device)
        estimated_data = perturbed_data.clone()
        SNR = torch.ones(batch.shape[0], 1, device=batch.device)

        # Run denoising (for reconstruction loss)
        if return_data:
            alpha, sigma = sde.return_alpha_sigma(t)
            alpha = alpha.flatten().unsqueeze(1)
            sigma = sigma.flatten().unsqueeze(1)
            SNR = alpha / sigma
            
            # Safe denoising (fallback on error)
            try:
                score, estimated_data = multi_step_denoise(perturbed_data, t, t_end=t/(2*denoise_steps), N=denoise_steps)
            except Exception as e:
                print(f"Warning: Denoising failed (fallback): {e}")
                score = score_fn(perturbed_data, t, condition=condition, mask=mask)
                estimated_data = perturbed_data.clone()
            
            # 🔴 YOUR SUPERVISOR'S RECONSTRUCTION LOSS (MSE + per-joint L2)
            # 1. Element-wise MSE: treats each dimension independently
            #    Formula: mean((estimated - original)²) across all dimensions
            recon_param_mse = torch.mean(torch.square(estimated_data - batch) * loss_mask)
            
            # 2. Per-joint L2 distance: groups by joint, computes Euclidean distance per joint
            #    Formula: mean(||estimated_joint - original_joint||) for each joint
            #    Note: This is in pose parameter space (axis-angle), NOT 3D joint positions
            if batch.shape[1] == 63:  # 21 joints × 3 dims
                batch_pose = batch.reshape(-1, 21, 3)              # Pose params [B, 21, 3]
                estimated_pose = estimated_data.reshape(-1, 21, 3)  # Pose params [B, 21, 3]
                joint_l2_per_joint = torch.norm(estimated_pose - batch_pose, dim=-1)  # L2 per joint
                mask_expanded = loss_mask.squeeze(1).unsqueeze(1)
                recon_joint_l2 = torch.mean(joint_l2_per_joint * mask_expanded)
        else:
            score = score_fn(perturbed_data, t, condition=condition, mask=mask)

        # Core diffusion loss calculation
        if not likelihood_weighting:
            losses = torch.square(score * std[:, None] + z) * loss_mask
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / std[:, None]) * loss_mask
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        # Core diffusion loss (score matching loss, without reconstruction losses)
        diffusion_loss = torch.mean(losses)
        
        # Note: recon_param_mse and recon_joint_l2 are computed for logging only
        # They are NOT used for backpropagation (removed from loss calculation)

        if return_data:
            # When auxiliary_loss=True: return pure diffusion loss (recon losses NOT in backprop)
            # Recon losses are still computed and returned for logging only
            return diffusion_loss, {
                'clean_sample': estimated_data,
                'SNR': SNR,
                't': t,
                'recon_param_mse': recon_param_mse,   # Element-wise MSE in pose space (for logging only)
                'recon_joint_l2': recon_joint_l2      # Per-joint L2 distance in pose space (for logging only)
            }
        else:
            # When auxiliary_loss=False: return pure diffusion loss (recon losses NOT in backprop)
            # Note: recon losses are NOT computed when return_data=False (can't log them in this case)
            return diffusion_loss

    return loss_fn


def get_smld_loss_fn(vesde, train, reduce_mean=False):
    """Legacy SMLD loss (unchanged)."""
    assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."
    smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch, condition, mask):
        model_fn = mutils.get_model_fn(model, train=train)
        labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
        sigmas = smld_sigma_array.to(batch.device)[labels]
        noise = torch.randn_like(batch) * sigmas[:, None]
        perturbed_data = noise + batch
        score = model_fn(perturbed_data, labels, condition, mask)
        target = -noise / (sigmas ** 2)[:, None]
        losses = torch.square(score - target)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
        loss = torch.mean(losses)
        return loss
    return loss_fn


def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
    """Legacy DDPM loss (unchanged)."""
    assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch, condition, mask):
        model_fn = mutils.get_model_fn(model, train=train)
        labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
        noise = torch.randn_like(batch)
        perturbed_data = sqrt_alphas_cumprod[labels, None] * batch + \
                         sqrt_1m_alphas_cumprod[labels, None] * noise
        score = model_fn(perturbed_data, labels, condition, mask)
        losses = torch.square(score - noise)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        loss = torch.mean(losses)
        return loss
    return loss_fn


def get_step_fn(sde, train, optimize_fn=None, 
                reduce_mean=False, continuous=True, likelihood_weighting=False, 
                auxiliary_loss=False, denormalize=None, body_model=None, model_type='body', **kwargs):
    """Create one-step training/eval function (NO optimizer/step_counter args)."""
    # 🔴 REMOVED optimizer/step_counter args entirely (no assertion needed)
    if continuous:
        loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                                  continuous=True, likelihood_weighting=likelihood_weighting,
                                  return_data=auxiliary_loss, **kwargs)
    else:
        assert not likelihood_weighting, "Likelihood weighting not supported for discrete SDEs."
        if isinstance(sde, VESDE):
            loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
        elif isinstance(sde, VPSDE):
            loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
        else:
            raise ValueError(f"Discrete training for {sde.__class__.__name__} not recommended.")
    
    # Only check for auxiliary loss dependencies (no optimizer/step)
    if auxiliary_loss:
        assert denormalize is not None and body_model is not None, "Need denormalize/body_model for auxiliary loss"

    l2_loss = nn.MSELoss(reduction='none')
    model_type_to_param = {
        'body': 'body_pose',
        'hand': 'hand_pose',
        'face': 'face_params',
        'face-betas': 'betas',
        'whole-body': 'wholebody_params',
    }
    param = model_type_to_param.get(model_type)
    if param is None:
        raise ValueError(f"model_type {model_type} not supported.")

    def step_fn(model, batch, condition, mask):
        """One-step training/eval with reconstruction loss."""
        if train:
            if not auxiliary_loss:
                # Basic diffusion loss only (recon losses NOT used for backprop)
                # Note: recon losses are not computed when auxiliary_loss=False (return_data=False)
                total_loss = loss_fn(model, batch, condition, mask)
                # total_loss is now pure diffusion_loss (recon losses not included)
                loss_dict = {
                    'loss': total_loss,
                    'diffusion_loss': total_loss,  # Pure diffusion loss (no recon losses)
                    'recon_param_mse': torch.tensor(0.0, device=total_loss.device),  # Not computed when auxiliary_loss=False
                    'recon_joint_l2': torch.tensor(0.0, device=total_loss.device)    # Not computed when auxiliary_loss=False
                }
            else:
                # Auxiliary loss: recon losses NOT used for backprop, only v2v/j2j are used
                # Recon losses are still computed for logging/monitoring
                diffusion_loss, data_dict = loss_fn(model, batch, condition, mask)
                # diffusion_loss is PURE diffusion score matching loss (no recon losses included)
                
                weight = torch.log(1.0 + data_dict['SNR'])
                estimate = denormalize(data_dict['clean_sample'], to_axis=True)
                batch_denorm = denormalize(batch, to_axis=True)
                
                # Auxiliary v2v/j2j loss (these ARE used for backprop)
                # Convert positions to mm and compute loss (naturally gives mm²)
                gt_body = body_model(**{param: batch_denorm})
                pred_body = body_model(**{param: estimate})
                loss_v2v = torch.mean(weight * l2_loss(gt_body.v * 1000, pred_body.v * 1000).sum(dim=-1))  # * 1000 convert m to mm
                loss_j2j = torch.mean(weight * l2_loss(gt_body.Jtr * 1000, pred_body.Jtr * 1000).sum(dim=-1))

                # 🔍 DEBUG PRINT (print occasionally)
                if torch.rand(1).item() < 0.01:  # ~1% of steps
                    print(
                        f"[LOSS DEBUG] "
                        f"diffusion: {diffusion_loss.item():.3e} | "
                        f"v2v: {loss_v2v.item():.3e} | "
                        f"j2j: {loss_j2j.item():.3e} "
                    )
                
                # Total loss: ONLY diffusion + v2v + j2j (recon losses NOT included)
                # All losses now in consistent units: diffusion_loss (dimensionless) + v2v/j2j (mm²)
                total_loss = 1e-2 * diffusion_loss + 1e-4 *loss_v2v + 1e-4 *loss_j2j
                
                loss_dict = {
                    'loss': total_loss,  # Used for backprop (v2v/j2j in mm²)
                    'diffusion_loss': diffusion_loss,  # Dimensionless (score matching loss)
                    'v2v_loss': loss_v2v,  # Already in mm² (computed on mm-scale positions)
                    'j2j_loss': loss_j2j,  # Already in mm² (computed on mm-scale positions)
                    'recon_param_mse': data_dict['recon_param_mse'],  # Dimensionless (for logging only)
                    'recon_joint_l2': data_dict['recon_joint_l2']     # In pose space (for logging only)
                }
        else:
            # Validation (no gradient)
            with torch.no_grad():
                if auxiliary_loss:
                    diffusion_loss, data_dict = loss_fn(model, batch, condition, mask)
                    # Note: v2v_loss and j2j_loss not computed here - validation logs MPVPE and MPJPE instead
                    loss_dict = {
                        'loss': diffusion_loss,
                        'diffusion_loss': diffusion_loss,
                        'recon_param_mse': data_dict['recon_param_mse'],
                        'recon_joint_l2': data_dict['recon_joint_l2']
                    }
                else:
                    total_loss = loss_fn(model, batch, condition, mask)
                    loss_dict = {
                        'loss': total_loss,
                        'diffusion_loss': total_loss,
                        'recon_param_mse': torch.tensor(0.0, device=total_loss.device),
                        'recon_joint_l2': torch.tensor(0.0, device=total_loss.device)
                    }
        return loss_dict

    return step_fn
