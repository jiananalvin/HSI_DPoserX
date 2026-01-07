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
        f"{args.name}_"
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

def log_dict_to_wandb(data_dict, step, prefix="train"):
    log_dict = {f"{prefix}/{k}": v for k, v in data_dict.items()}
    wandb.log(log_dict, step=step)