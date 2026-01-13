"""
Training visualization utilities for pose diffusion models.

This module provides functions for rendering and logging pose visualizations
during training, including text overlay and grid creation.
"""

import os
import numpy as np
import torch
import torchvision
import cv2
import wandb

from lib.body_model.visual import render_mesh


def convert_to_tensor(img):
    """
    Convert numpy image to torch tensor for TensorBoard logging.
    
    Args:
        img: numpy array [H, W, 3] or [H, W]
    
    Returns:
        torch tensor [C, H, W] in [0, 1]
    """
    img = img.astype(np.float32)
    # Normalize only if image is 0–255
    if img.max() > 1.0:
        img = img / 255.0

    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.permute(2, 0, 1)

    return img_tensor


def overlay_text_on_image(rendered_img, text_prompts, tag_prefix, idx, sample_idx=None, step_idx=None):
    """
    Overlay text prompts on rendered images.
    
    Args:
        rendered_img: numpy array [H, W, 3] - rendered mesh image
        text_prompts: list of strings - text prompts to overlay
        tag_prefix: str - 'trajs' or 'samples'
        idx: int - current image index
        sample_idx: int - sample index (for grouping)
        step_idx: int - step index (for trajectories)
    
    Returns:
        numpy array [H, W, 3] - image with text overlay
    """
    if text_prompts is None:
        return rendered_img
    
    # Determine sample index
    if sample_idx is None:
        sample_idx = idx // 10  # 0-4 for 5 samples
    
    if sample_idx >= len(text_prompts):
        return rendered_img
    
    prompt = text_prompts[sample_idx]
    # Split into 2 lines (53 chars per line)
    line1 = prompt[:53]
    line2 = prompt[53:106] if len(prompt) > 53 else ""
    
    # Font config
    font_size = 0.4
    font_thickness_outline = 2
    font_thickness_text = 1
    text_color_outline = (0, 0, 0)  # Black
    text_color = (255, 255, 255)    # White
    
    def draw_text_with_outline(img, text, x, y):
        """Draw text with outline for better visibility."""
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                   font_size, text_color_outline, font_thickness_outline, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                   font_size, text_color, font_thickness_text, cv2.LINE_AA)
    
    # Line 1 of prompt (top)
    draw_text_with_outline(rendered_img, line1, 10, 25)
    
    # Line 2 of prompt (middle)
    if line2:
        draw_text_with_outline(rendered_img, line2, 10, 50)
    
    # For trajs_grid: add step number (3rd line, bottom)
    if tag_prefix == 'trajs' and step_idx is not None:
        step_text = f"Step: {step_idx+1}/10"
        draw_text_with_outline(rendered_img, step_text, 10, 75)
    
    return rendered_img


def process_and_log_meshes(poses, bg_img, focal, princpt, tag_prefix,
                           body_model_vis, logger, global_step, current_epoch,
                           text_prompts=None, pose_dim=3, nrow=10):
    """
    Process poses into meshes, render, overlay text, and log to TensorBoard/wandb.
    
    Args:
        poses: torch.Tensor [B, pose_dim] - pose parameters
        bg_img: numpy array [H, W, 3] - background image
        focal: list [fx, fy] - camera focal length
        princpt: list [cx, cy] - camera principal point
        tag_prefix: str - prefix for logging tag ('trajs' or 'samples')
        body_model_vis: BodyModel - body model for mesh generation
        logger: PyTorch Lightning logger - for TensorBoard logging
        global_step: int - current training step
        current_epoch: int - current epoch
        text_prompts: list of str - optional text prompts to overlay
        pose_dim: int - pose dimension (3 for axis-angle, 6 for rotation matrix)
        nrow: int - number of columns in grid
    
    Returns:
        numpy array [H, W, 3] - image grid as numpy array
    """
    try:
        # Validate pose shape
        if poses.dim() == 2:
            batch_size, pose_dim_actual = poses.shape
            # Handle both 63D (body_pose only) and 66D (global_orient + body_pose)
            if pose_dim_actual == 66:
                # Extract body_pose only (skip first 3 dims which are global_orient)
                poses = poses[:, 3:]
            elif pose_dim_actual != 63:
                print(f"⚠️ Pose dim mismatch: expected 63 or 66, got {pose_dim_actual} — truncating/padding!")
                if pose_dim_actual > 63:
                    poses = poses[:, :63]
                else:
                    poses = torch.cat([poses, torch.zeros(batch_size, 63 - pose_dim_actual, device=poses.device)], dim=-1)
        
        # Generate SMPL-X meshes
        body_out = body_model_vis(body_pose=poses)
        meshes = body_out.v.detach().cpu().numpy()
        faces = body_out.f.cpu().numpy()
        
        # Render each mesh + overlay text prompts
        rendered_images = []
        for idx, mesh in enumerate(meshes):
            rendered_img = render_mesh(bg_img, mesh, faces, {'focal': focal, 'princpt': princpt})
            if rendered_img is None:
                rendered_img = np.ones_like(bg_img) * 255
            
            # Overlay text prompts
            if text_prompts is not None:
                sample_idx = idx // nrow  # 0-4 for 5 samples (with nrow=10)
                step_idx = idx % nrow if tag_prefix == 'trajs' else None
                rendered_img = overlay_text_on_image(
                    rendered_img, text_prompts, tag_prefix, idx, 
                    sample_idx=sample_idx, step_idx=step_idx
                )
            
            # Convert to tensor
            rendered_img_tensor = convert_to_tensor(rendered_img)
            rendered_images.append(rendered_img_tensor)
        
        # Create image grid
        image_grid = torchvision.utils.make_grid(rendered_images, nrow=nrow)
        
        # Log to TensorBoard
        logger.experiment.add_image(f'{tag_prefix}_grid', image_grid, current_epoch)
        
        # Convert to numpy and save
        image_grid_np = image_grid.permute(1, 2, 0).cpu().numpy()
        image_grid_np = (image_grid_np * 255).astype(np.uint8)
        
        os.makedirs(f"logs/renders/step_{global_step}", exist_ok=True)
        cv2.imwrite(f"logs/renders/step_{global_step}/{tag_prefix}_grid.png", image_grid_np[:, :, ::-1])
        print(f"✅ Saved {tag_prefix}_grid.png to logs/renders/step_{global_step}/")
        
        return image_grid_np
    
    except Exception as e:
        print(f"❌ process_and_log_meshes failed for {tag_prefix}: {e}")
        import traceback
        traceback.print_exc()
        dummy_img = np.ones((512, 384, 3), dtype=np.uint8) * 255
        return dummy_img


def render_and_log_images(trajs, all_results, denormalize_fn, sde,
                          body_model_vis, logger, global_step, current_epoch,
                          text_prompts=None, pose_dim=3):
    """
    Render trajectories and samples for training visualization.
    
    Args:
        trajs: torch.Tensor [T, B, pose_dim] - diffusion trajectories
        all_results: torch.Tensor [B, pose_dim] - final generated samples
        denormalize_fn: callable - function to denormalize poses
        sde: SDE object - stochastic differential equation
        body_model_vis: BodyModel - body model for visualization
        logger: PyTorch Lightning logger - for TensorBoard logging
        global_step: int - current training step
        current_epoch: int - current epoch
        text_prompts: list of str - optional text prompts to overlay
        pose_dim: int - pose dimension (3 for axis-angle, 6 for rotation matrix)
    
    Returns:
        tuple: (traj_grid, sample_grid) - numpy arrays of image grids
    """
    bg_img = np.ones([512, 384, 3]) * 255
    focal = [1500, 1500]
    princpt = [200, 192]
    
    # Sample frames for visualization
    slice_step = sde.N // 10
    trajs_processed = denormalize_fn(trajs[::slice_step, :5, ], to_axis=True).reshape(50, -1)
    all_results_processed = denormalize_fn(all_results[:50], to_axis=True)
    
    # Process and log trajectories
    traj_grid = process_and_log_meshes(
        trajs_processed, bg_img, focal, princpt, 'trajs',
        body_model_vis, logger, global_step, current_epoch,
        text_prompts=text_prompts, pose_dim=pose_dim
    )
    
    # Process and log samples
    sample_grid = process_and_log_meshes(
        all_results_processed, bg_img, focal, princpt, 'samples',
        body_model_vis, logger, global_step, current_epoch,
        text_prompts=text_prompts, pose_dim=pose_dim
    )
    
    # Log to wandb
    wandb.log({"val/trajs_grid": wandb.Image(traj_grid)}, step=global_step)
    wandb.log({"val/samples_grid": wandb.Image(sample_grid)}, step=global_step)
    
    return traj_grid, sample_grid
