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

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods (modified for text-to-3D pose generation)."""
import functools
import inspect

import torch
import numpy as np
import abc

from .utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
from . import sde_lib
from . import utils as mutils

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps, device=None, inverse_solver=None):
    """Create a sampling function (supports text condition).

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.
    device: PyTorch device.
    inverse_solver: A `str` or `None`. The inverse problem solver used in the predictor.

  Returns:
    A function that takes random states, text condition, and a model and outputs text-conditioned 3D pose samples.
  """
    if device is None:
        device = config.device
    sampler_name = config.sampling.method
    # Probability flow ODE sampling with black-box ODE solvers
    if sampler_name.lower() == 'ode':
        sampling_fn = get_ode_sampler(sde=sde,
                                      shape=shape,
                                      inverse_scaler=inverse_scaler,
                                      denoise=config.sampling.noise_removal,
                                      eps=eps,
                                      device=device)
    # Predictor-Corrector sampling (main sampler for DPoser)
    elif sampler_name.lower() == 'pc':
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        sampling_fn = get_pc_sampler(sde=sde,
                                     shape=shape,
                                     predictor=predictor,
                                     corrector=corrector,
                                     inverse_scaler=inverse_scaler,
                                     snr=config.sampling.snr,
                                     n_steps=config.sampling.n_steps_each,
                                     probability_flow=config.sampling.probability_flow,
                                     continuous=config.training.continuous,
                                     denoise=config.sampling.noise_removal,
                                     inverse_solver=inverse_solver,
                                     eps=eps,
                                     device=device)
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm (supports text condition)."""

    def __init__(self, sde, score_fn, probability_flow=False, inverse_solver=None):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t, condition, mask):
        """One update of the predictor (with text condition).

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.
      condition: [B, 768] (CLIP text embeddings, optional)
      mask: Mask for completion (optional)

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm (supports text condition)."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t, condition, mask):
        """One update of the corrector (with text condition).

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.
      condition: [B, 768] (CLIP text embeddings, optional)
      mask: Mask for completion (optional)

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


# Note: only euler_maruyama predictor used in our experiments
@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False, inverse_solver=None):
        super().__init__(sde, score_fn, probability_flow, inverse_solver)
        self.inverse_solver = inverse_solver

    def update_fn(self, x, t, condition, mask, observation=None, grad_step=1.0):  # Added observation=None for safety
        # Fix 1: Move dt to x's device (CUDA)
        dt = torch.tensor(-1. / self.rsde.N, device=x.device, dtype=x.dtype)
        z = torch.randn_like(x)
        
        if observation is not None:
            if self.inverse_solver == 'BP':
                x.requires_grad_()
                drift, diffusion, alpha, sigma_2, score = self.rsde.sde(x, t, condition=condition, mask=None, guide=True)
                
                # Fix 2: Ensure all tensors are on CUDA
                drift = drift.to(x.device)
                diffusion = diffusion.to(x.device)
                alpha = alpha.to(x.device)
                sigma_2 = sigma_2.to(x.device)
                score = score.to(x.device)
                
                y_t_mean = x.detach() + drift.detach() * dt
                y_t_hat = y_t_mean + diffusion[:, None] * torch.sqrt(-dt) * z

                with torch.enable_grad():  # Enable gradients computation
                    y_0_hat = (x + sigma_2[:, None] * score) / alpha
                    norm = torch.norm((observation * mask) - (y_0_hat * mask))
                    norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
                    if torch.isnan(norm_grad).any():
                        raise ValueError('Consider reduce the value of parameter: grad_step={}'.format(grad_step))
                    y_t_hat = y_t_hat - grad_step * norm_grad

                return y_t_hat, y_t_mean

            elif self.inverse_solver == 'ABP':  # Adaptive BP from DSG
                x.requires_grad_()
                drift, diffusion, alpha, sigma_2, score = self.rsde.sde(x, t, condition=condition, mask=None, guide=True)
                
                # Fix 3: Ensure all tensors are on CUDA
                drift = drift.to(x.device)
                diffusion = diffusion.to(x.device)
                alpha = alpha.to(x.device)
                sigma_2 = sigma_2.to(x.device)
                score = score.to(x.device)
                
                y_t_mean = x.detach() + drift.detach() * dt
                y_t_hat = y_t_mean + diffusion[:, None] * torch.sqrt(-dt) * z

                with torch.enable_grad():  # Enable gradients computation
                    y_0_hat = (x + sigma_2[:, None] * score) / alpha
                    norm = torch.norm((observation * mask) - (y_0_hat * mask))
                    grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
                    if torch.isnan(grad).any():
                        raise ValueError('Consider reduce the value of parameter: grad_step={}'.format(grad_step))
                    grad_norm = torch.linalg.norm(grad, dim=[1])[:, None]
                    b, c, = x.shape
                    r = torch.sqrt(torch.tensor(c, device=x.device)) * (diffusion * torch.sqrt(-dt))[0]
                    guidance_rate, eps = 1.0, 1e-8

                    d_star = -r * grad / (grad_norm + eps)
                    d_sample = y_t_hat - y_t_mean
                    mix_direction = d_sample + guidance_rate * (d_star - d_sample)
                    mix_direction_norm = torch.linalg.norm(mix_direction, dim=[1])[:, None]
                    mix_step = mix_direction / (mix_direction_norm + eps) * r

                return y_t_mean + mix_step, y_t_mean

        # Fix 4: Get drift/diffusion and move to CUDA
        drift, diffusion = self.rsde.sde(x, t, condition=condition, mask=mask)
        drift = drift.to(x.device)
        diffusion = diffusion.to(x.device)
        
        # Fix 5: All operations on CUDA
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None] * torch.sqrt(-dt) * z
        
        return x, x_mean


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False, inverse_solver=None):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t, condition=None, mask=None):
        f, G = self.rsde.discretize(x, t, condition=condition, mask=mask)
        
        # Fix: Move f/G to CUDA
        f = f.to(x.device)
        G = G.to(x.device)
        
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None] * z
        return x, x_mean


@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
    """The ancestral sampling predictor (supports text condition). Currently only supports VE/VP SDEs."""

    def __init__(self, sde, score_fn, probability_flow=False, inverse_solver=None):
        super().__init__(sde, score_fn, probability_flow)
        if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
        assert not probability_flow, "Probability flow not supported by ancestral sampling"

    def vesde_update_fn(self, x, t, condition, mask):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        sigma = sde.discrete_sigmas.to(x.device)[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
        score = self.score_fn(x, t, condition=condition, mask=mask)
        
        # Fix: Move score to CUDA
        score = score.to(x.device)
        
        x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None]
        std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
        noise = torch.randn_like(x)
        x = x_mean + std[:, None] * noise
        return x, x_mean

    def vpsde_update_fn(self, x, t, condition, mask):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        beta = sde.discrete_betas.to(t.device)[timestep]
        score = self.score_fn(x, t, condition=condition, mask=mask)
        
        # Fix: Move score to CUDA
        score = score.to(x.device)
        
        x_mean = (x + beta[:, None] * score) / torch.sqrt(1. - beta)[:, None]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None] * noise
        return x, x_mean

    def update_fn(self, x, t, condition=None, mask=None):
        if isinstance(self.sde, sde_lib.VESDE):
            return self.vesde_update_fn(x, t, condition=condition, mask=mask)
        elif isinstance(self.sde, sde_lib.VPSDE):
            return self.vpsde_update_fn(x, t, condition=condition, mask=mask)


@register_predictor(name='none')
class NonePredictor(Predictor):
    """An empty predictor that does nothing (supports text condition)."""

    def __init__(self, sde, score_fn, probability_flow=False, inverse_solver=None):
        super().__init__(sde, score_fn, probability_flow, inverse_solver)

    def update_fn(self, x, t, condition=None, mask=None):
        return x, x


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t, condition=None, mask=None):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        
        # Fix: Move alpha to CUDA
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(x.device)[timestep]
        else:
            alpha = torch.ones_like(t, device=x.device)

        for i in range(n_steps):
            grad = score_fn(x, t, condition=condition, mask=mask)
            
            # Fix: Move grad to CUDA
            grad = grad.to(x.device)
            
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            
            # Fix: Move step_size to CUDA
            step_size = step_size.to(x.device)
            
            x_mean = x + step_size[:, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None] * noise

        return x, x_mean


@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor (supports text condition)."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t, condition=None, mask=None):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        
        # Fix: Move alpha to CUDA
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(x.device)[timestep]
        else:
            alpha = torch.ones_like(t, device=x.device)

        # Fix: Move std to CUDA
        std = self.sde.marginal_prob(x, t)[1].to(x.device)

        for i in range(n_steps):
            grad = score_fn(x, t, condition=condition, mask=mask)
            
            # Fix: Move grad to CUDA
            grad = grad.to(x.device)
            
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            
            # Fix: Move step_size to CUDA
            step_size = step_size.to(x.device)
            
            x_mean = x + step_size[:, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None]

        return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
    """An empty corrector that does nothing (supports text condition)."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)

    def update_fn(self, x, t, condition=None, mask=None):
        return x, x


def shared_predictor_update_fn(x, t, condition, mask, sde, model, observation, predictor,
                               probability_flow, continuous, inverse_solver=None):
    """A wrapper that configures and returns the update function of predictors (supports text condition)."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow, inverse_solver)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow, inverse_solver)

    if 'observation' in inspect.signature(predictor_obj.update_fn).parameters:
        return predictor_obj.update_fn(x, t, condition=condition, mask=mask, observation=observation)
    else:
        return predictor_obj.update_fn(x, t, condition=condition, mask=mask)


def shared_corrector_update_fn(x, t, condition, mask, sde, model, observation, corrector, continuous, snr, n_steps):
    """A wrapper that configures and returns the update function of correctors (supports text condition)."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(x, t, condition=condition, mask=mask)


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, inverse_solver=None, eps=1e-3, device='cuda'):
    """Create a Predictor-Corrector (PC) sampler (supports text-to-3D pose generation).

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    inverse_solver: A `str` or `None`. The inverse problem solver used in the predictor.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns text-conditioned 3D pose samples.
  """
    # Create predictor & corrector update functions
    assert inverse_solver in [None, 'BP', 'ABP'], f"Unknown inverse solver {inverse_solver}"
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous,
                                            inverse_solver=inverse_solver)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def get_imputation_update_fn(update_fn):
        """Modify the update function to incorporate text condition and data information."""

        def imputation_update_fn(x, vec_t, observation, condition, mask, model, args):
            # Fix: Move all inputs to CUDA
            x = x.to(device)
            vec_t = vec_t.to(device)
            if observation is not None:
                observation = observation.to(device)
            if mask is not None:
                mask = mask.to(device)
            if condition is not None:
                condition = condition.to(device)
                
            x, x_mean = update_fn(x, vec_t, condition=condition, mask=mask, model=model, observation=observation)

            if args is not None and args.task in ['completion']:
                masked_data_mean, std = sde.marginal_prob(observation, vec_t)
                
                # Fix: Move masked_data_mean/std to CUDA
                masked_data_mean = masked_data_mean.to(x.device)
                std = std.to(x.device)
                
                masked_data = masked_data_mean + torch.randn_like(x) * std[:, None]

                x = x * (~mask) + masked_data * mask

            return x, x_mean

        return imputation_update_fn

    projector_imputation_update_fn = get_imputation_update_fn(predictor_update_fn)
    corrector_imputation_update_fn = get_imputation_update_fn(corrector_update_fn)

    def pc_sampler(model, observation=None, condition=None, mask=None, z=None, start_step=0, args=None, gather_traj=True):
        """ The PC sampler for text-to-3D pose generation.
    Args:
      model: A score model (with text cross-attention).
      condition: [B, 768] (CLIP text embeddings for text conditioning).
      observation: partial information for completion (optional).
      mask: mask for completion (optional).
      z: initial noise for denoising (optional).
      start_step: intermediate timestep for denoising (optional).
      args: task description (optional).
      gather_traj: if True, gather intermediate trajectories for debugging.

    Returns:
      trajs: Intermediate sampling trajectories (optional).
      x_mean/x: Text-conditioned 3D pose samples.
    """
        with torch.no_grad():
            # Initial sample (noise from SDE prior)
            if z is None:
                x = sde.prior_sampling(shape).to(device)
            else:
                x = z.to(device)  # Fix: Move z to CUDA
                
            # Fix: Move timesteps to CUDA
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
            trajs = []

            start_t = 0
            if args is not None and args.task in ['denoise', 'debug']:
                start_t = start_step

            # Reverse diffusion loop (text-conditioned)
            for i in range(start_t, sde.N):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                
                # Fix: Move vec_t to CUDA
                vec_t = vec_t.to(device)
                
                # Corrector step (text-conditioned)
                x, x_mean = corrector_imputation_update_fn(x, vec_t, observation, condition, mask, model=model,
                                                           args=args)
                # Predictor step (text-conditioned)
                x, x_mean = projector_imputation_update_fn(x, vec_t, observation, condition, mask, model=model,
                                                           args=args)
                if gather_traj:
                    trajs.append(x)

            trajs = torch.stack(trajs, dim=0) if gather_traj else None  # [t, b, j*3]

            # Return text-conditioned 3D pose samples
            return trajs, x_mean if denoise else x

    return pc_sampler


def get_ode_sampler(sde, shape, inverse_scaler,
                    denoise=False, rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda'):
    """Probability flow ODE sampler (supports text-to-3D pose generation).

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

  Returns:
    A sampling function that returns text-conditioned 3D pose samples.
  """

    def denoise_update_fn(model, x, condition=None):  # Added condition for text denoising
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        # Reverse diffusion predictor for denoising (text-conditioned)
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor_obj.update_fn(x, vec_eps, condition=condition, mask=None)
        return x

    def drift_fn(model, x, t, condition=None):  # Added condition for text conditioning
        """Get the drift function of the reverse-time SDE (text-conditioned)."""
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t, condition=condition, mask=None)[0]

    def ode_sampler(model, z=None, condition=None):  # Added condition for text input
        """The probability flow ODE sampler for text-to-3D pose generation.

    Args:
      model: A score model (with text cross-attention).
      z: If present, generate samples from latent code `z` (optional).
      condition: [B, 768] (CLIP text embeddings for text conditioning).

    Returns:
      nfe: Number of function evaluations.
      x: Text-conditioned 3D pose samples.
    """
        with torch.no_grad():
            # Initial sample (noise from SDE prior)
            if z is None:
                x = sde.prior_sampling(shape).to(device)
            else:
                x = z.to(device)  # Fix: Move z to CUDA

            def ode_func(t, x):
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = drift_fn(model, x, vec_t, condition=condition)
                return to_flattened_numpy(drift)

            # Black-box ODE solver (text-conditioned)
            solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                           rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

            # Denoising (text-conditioned)
            if denoise:
                x = denoise_update_fn(model, x, condition=condition)

            x = inverse_scaler(x)
            return nfe, x

    return ode_sampler