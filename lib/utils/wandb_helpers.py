# wandb_helpers.py
import wandb
import torch
import torchvision
import numpy as np
from PIL import Image
import os
from lib.body_model.visual import render_mesh

def init_wandb(config, args):
    """Initialize wandb run with project/config metadata + meaningful names."""
    run_name = (
        f"{config.training.sde}_"
        f"lr{config.optim.lr}_"
        f"bs{config.training.batch_size}_"
        f"{config.training.n_iters/1000}k_steps_"
        f"{config.data.rot_rep}_"
        f"ema{config.model.ema_rate}"
    )
    
    wandb.init(
        project="text-to-3d-pose-diffusion",
        name=run_name,  # Use dynamic/meaningful name
        config={
            "dataset": config.dataset,
            "sde_type": config.training.sde,
            "lr": config.optim.lr,
            "batch_size": config.training.batch_size,
            "num_scales": config.model.num_scales,
            "ema_rate": config.model.ema_rate,
            "max_steps": config.training.n_iters,
            "rot_rep": config.data.rot_rep,
            "normalize": config.data.normalize,
        },
        save_code=True,
    )
    return wandb

def log_losses(loss_dict, step, prefix="train"):
    """Log loss values to wandb (supports train/val prefixes)."""
    log_dict = {f"{prefix}/{k}": v.item() for k, v in loss_dict.items()}
    wandb.log(log_dict, step=step)

def create_noise_denoise_gif(
    model, sde, normalize_fn, denormalize_fn, 
    body_model_vis, sample_pose, text_embed, 
    num_steps=50, device="cuda", save_path="noise_denoise.gif"
):
    """
    Create a GIF showing:
    1. Adding noise to a clean pose (forward SDE)
    2. Denoising the noisy pose (reverse SDE/text-conditioned)
    """
    # Step 1: Add noise to clean pose (forward SDE)
    clean_pose = normalize_fn(sample_pose, from_axis=True).unsqueeze(0)  # [1, 63]
    noise_steps = torch.linspace(0, sde.T, num_steps, device=device)
    noisy_poses = []
    
    for t in noise_steps:
        t_vec = torch.ones(1, device=device) * t
        mean, std = sde.marginal_prob(clean_pose, t_vec)
        noisy_pose = mean + std * torch.randn_like(clean_pose)
        noisy_poses.append(denormalize_fn(noisy_pose, to_axis=True))
    
    # Step 2: Denoise the final noisy pose (reverse SDE/text-conditioned)
    final_noisy_pose = noisy_poses[-1]
    denoise_steps = []
    
    # Get text-conditioned sampling function (use PC sampler)
    sampling_fn = model.sampling_fn
    trajs, _ = sampling_fn(
        model.model,
        observation=None,
        condition=text_embed,
        z=normalize_fn(final_noisy_pose, from_axis=True).unsqueeze(0),
        start_step=0,
        gather_traj=True
    )
    
    # Convert trajs to denormalized poses
    trajs = trajs[:, 0, :]  # [num_steps, 63]
    for traj in trajs:
        denoise_steps.append(denormalize_fn(traj.unsqueeze(0), to_axis=True))
    
    # Step 3: Render all frames (noise + denoise)
    all_frames = noisy_poses + denoise_steps
    rendered_frames = []
    
    # Render settings
    bg_img = np.ones([512, 384, 3]) * 255
    focal = [1500, 1500]
    princpt = [200, 192]
    
    for pose in all_frames:
        body_out = body_model_vis(body_pose=pose)
        mesh = body_out.v.detach().cpu().numpy()[0]
        faces = body_out.f.cpu().numpy()
        
        # Render mesh
        img = render_mesh(bg_img, mesh, faces, {'focal': focal, 'princpt': princpt})
        img_pil = Image.fromarray(img.astype(np.uint8))
        rendered_frames.append(img_pil)
    
    # Step 4: Save GIF
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    rendered_frames[0].save(
        save_path,
        save_all=True,
        append_images=rendered_frames[1:],
        duration=100,  # 100ms per frame
        loop=0
    )
    
    # Log GIF to wandb
    wandb.log({"noise_denoise_gif": wandb.Video(save_path, format="gif")})
    return save_path

def log_metrics(metrics, step, prefix="val"):
    """Log validation metrics (APD/SI/MPVPE/MPJPE/BPD) to wandb."""
    log_dict = {f"{prefix}/{k}": v for k, v in metrics.items()}
    wandb.log(log_dict, step=step)

def log_rendered_images(image_grid, step, tag):
    """Log rendered pose grids to wandb."""
    wandb.log({tag: wandb.Image(image_grid)}, step=step)