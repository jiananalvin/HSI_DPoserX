import os
import sys
import argparse

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torchvision
from torch import optim
import cv2

from lib.body_model.body_model import BodyModel
from lib.utils.callbacks import TimerCallback, ModelSizeCallback
from lib.utils.misc import create_mask
from lib.utils.metric import average_pairwise_distance, self_intersections_percentage
from lib.utils.generic import import_configs

from lib.dataset.body import AMASSDataModule, N_POSES, Evaler
from lib.dataset.utils import Posenormalizer

from lib.algorithms.advanced.model import create_model
from lib.algorithms.advanced import losses, sde_lib, sampling, likelihood
from lib.algorithms.ema import ExponentialMovingAverage
from lib.algorithms.completion import DPoserComp
from lib.utils.schedulers import CosineWarmupScheduler

import open_clip
import wandb
from lib.utils.wandb_helpers import init_wandb, log_dict_to_wandb
from lib.utils.training_visualization import render_and_log_images
from lib.utils.text_encoders import create_text_encoder, RECOMMENDED_ENCODERS
from lib.utils.checkpoint_utils import load_pretrained_checkpoint_for_finetuning, set_requires_grad_for_text_conditioning_only

def parse_args(argv):
    parser = argparse.ArgumentParser(description='train diffusion model')
    parser.add_argument('--config-path', '-c', type=str,
                        default='configs.body.subvp.timefc.get_config',
                        help='config files to build DPoser')
    parser.add_argument('--bodymodel-path', type=str,
                        default='./body_models/smplx/SMPLX_NEUTRAL.npz',
                        help='load SMPLX for visualization')
    parser.add_argument('--resume-ckpt', '-r', type=str, help='resume training')
    parser.add_argument('--pretrained-ckpt', type=str, help='load pretrained checkpoint (without text conditioning) for fine-tuning')
    parser.add_argument('--freeze-pretrained', action='store_true', help='freeze pretrained parameters (only train text conditioning)')
    parser.add_argument('--data-root', type=str,
                        default='./data/body_data', help='dataset root')
    parser.add_argument('--version', type=str, default='version1', help='dataset version')
    parser.add_argument('--sample', type=int, help='sample trainset to reduce data')
    parser.add_argument('--name', type=str, default='default', help='name of checkpoint folder')

    args = parser.parse_args(argv[1:])

    return args


class DPoserTrainer(pl.LightningModule):
    def __init__(self, config,
                 bodymodel_path='',
                 data_path='',
                 N_POSES=21,
                 train_loader=None,
                 val_loader=None,
                 pretrained_ckpt=None,
                 freeze_pretrained=False):
        super().__init__()
        self.config = config
        self.bodymodel_path = bodymodel_path
        self.data_path = data_path
        self.N_POSES = N_POSES
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.pretrained_ckpt = pretrained_ckpt
        self.freeze_pretrained = freeze_pretrained
        self.save_hyperparameters(ignore=['train_loader', 'val_loader'])

        # Text Encoder - configurable via config or default to CLIP for backward compatibility
        text_encoder_config = getattr(config, 'text_encoder', None)
        if text_encoder_config is not None:
            encoder_type = getattr(text_encoder_config, 'type', 'clip')
            encoder_model = getattr(text_encoder_config, 'model_name', None)
        else:
            encoder_type = 'clip'
            encoder_model = None
        
        # Create text encoder (will be initialized in setup() when device is available)
        self.text_encoder = None  # Will be initialized in setup() when device is available
        self.text_encoder_type = encoder_type
        self.text_encoder_model = encoder_model
        
        # Get embedding dimension based on encoder type and model
        # Handle special cases for model-specific dimensions
        if encoder_type == 'sentence-bert' and encoder_model == 'all-MiniLM-L6-v2':
            self.text_embedding_dim = 384
        elif encoder_type == 't5' and encoder_model == 't5-small':
            self.text_embedding_dim = 512
        elif encoder_type in RECOMMENDED_ENCODERS:
            _, _, self.text_embedding_dim = RECOMMENDED_ENCODERS[encoder_type]
        else:
            # Default to CLIP dimensions for backward compatibility
            self.text_embedding_dim = 768

        # Collect data
        self.last_trajs = None
        self.all_samples = []

        # Diffusion model
        self.POSE_DIM = 3 if config.data.rot_rep == 'axis' else 6
        self.model = create_model(config.model, N_POSES, self.POSE_DIM, text_embedding_dim=self.text_embedding_dim)
        self.model_ema = None

        # Body models
        self.body_model_vis = BodyModel(
            bm_path=self.bodymodel_path,
            num_betas=10,
            batch_size=config.eval.batch_size,
            model_type='smplx'
        )
        self.body_model_eval = BodyModel(
            bm_path=self.bodymodel_path,
            num_betas=10,
            batch_size=config.eval.batch_size,
            model_type='smplx'
        )
        for p in self.body_model_vis.parameters():
            p.requires_grad = False
        for p in self.body_model_eval.parameters():
            p.requires_grad = False

        self.normalize_fn = None
        self.denormalize_fn = None

        # SDE & sampling
        self.sde = self.setup_sde(config)
        self.sampling_shape = (config.eval.batch_size, N_POSES * self.POSE_DIM)
        self.sampling_eps = 1e-3
        self.train_step_fn = None
        self.sampling_fn = None
        self.likelihood_fn = likelihood.get_likelihood_fn(
            self.sde, lambda x: x, rtol=1e-4, atol=1e-4, eps=1e-4
        )
        self.compfn = None
        self.last_text_prompts = None
    
    @torch.no_grad()
    def encode_text(self, text_list):
        """Encode text using the configured text encoder."""
        if self.text_encoder is None:
            raise RuntimeError("Text encoder not initialized. Call setup() first.")
        return self.text_encoder.encode(text_list)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def setup(self, stage=None):
        if stage == 'fit':
            # Initialize text encoder (needs device to be set)
            if self.text_encoder is None:
                print(f"📝 Initializing text encoder: {self.text_encoder_type}")
                if self.text_encoder_model:
                    print(f"   Model: {self.text_encoder_model}")
                self.text_encoder = create_text_encoder(
                    encoder_type=self.text_encoder_type,
                    model_name=self.text_encoder_model,
                    device=self.device
                )
                self.text_embedding_dim = self.text_encoder.embedding_dim
                print(f"   Embedding dimension: {self.text_embedding_dim}")
                print(f"   Max text length: {self.text_encoder.max_length} tokens")
                
                # Update model with correct text embedding dimension if needed
                # (Only if embedding dim changed from default 768)
                if self.text_embedding_dim != 768:
                    print(f"⚠️  Warning: Text embedding dim is {self.text_embedding_dim}, not 768.")
                    print(f"   Make sure your model was initialized with text_embedding_dim={self.text_embedding_dim}")
            
            self.body_model_vis = self.body_model_vis.to(self.device)
            self.body_model_eval = self.body_model_eval.to(self.device)
            self.model_ema = ExponentialMovingAverage(self.model.parameters(),
                                                      decay=self.config.model.ema_rate,
                                                      device=self.device)
            
            # Load pretrained checkpoint if provided (for fine-tuning from non-text model)
            if self.pretrained_ckpt is not None:
                print(f"🔄 Loading pretrained checkpoint from: {self.pretrained_ckpt}")
                print(f"   This checkpoint doesn't have text conditioning - will initialize text layers randomly")
                stats = load_pretrained_checkpoint_for_finetuning(
                    self.model,
                    self.pretrained_ckpt,
                    device=self.device,
                    strict=False,  # Allow missing keys (text conditioning layers)
                    is_ema=True,
                    freeze_pretrained=self.freeze_pretrained
                )
                print(f"✅ Pretrained checkpoint loaded: {stats['loaded']} params loaded, {stats['missing']} missing (text conditioning)")
                
                if self.freeze_pretrained:
                    print("🔒 Pretrained parameters are frozen - only text conditioning will be trained")
                    print("   (Use this for two-stage fine-tuning: first train text layers, then unfreeze)")
                else:
                    print("🔓 All parameters are trainable (pretrained + text conditioning)")
                    print("   (Recommended: fine-tune everything together)")
            
            Normalizer = Posenormalizer(
                data_path=self.data_path,
                normalize=self.config.data.normalize,
                min_max=self.config.data.min_max,
                rot_rep=self.config.data.rot_rep,
                device=self.device
            )
            self.normalize_fn = Normalizer.offline_normalize
            self.denormalize_fn = Normalizer.offline_denormalize
            self.train_step_fn = self.setup_step_fn(self.config)
            self.val_step_fn = self.setup_step_fn_val(self.config)
            self.sampling_fn = sampling.get_sampling_fn(self.config, self.sde, self.sampling_shape,
                                                        lambda x: x, self.sampling_eps, self.device)


    def setup_step_fn(self, config):
        # Build one-step training and evaluation functions
        # NOTE: For text-to-pose training, auxiliary_loss=True is recommended to enable
        # reconstruction loss (recon_param_mse/recon_joint_l2) which helps the model learn
        # to reconstruct poses from text embeddings.
        kwargs = {}
        if config.training.auxiliary_loss:
            body_model_train = BodyModel(bm_path=self.bodymodel_path,
                                         num_betas=10,
                                         batch_size=config.training.batch_size,
                                         model_type='smplx').to(self.device)
            for param in body_model_train.parameters():
                param.requires_grad = False
            aux_params = {'denormalize': self.denormalize_fn, 'body_model': body_model_train,
                          'model_type': "body", 'denoise_steps': config.training.denoise_steps}
            kwargs.update(aux_params)
        if config.training.random_mask:
            mask_params = {'min_mask_rate': config.training.min_mask_rate,
                           'max_mask_rate': config.training.max_mask_rate,
                           'observation_type': config.training.observation_type}
            kwargs.update(mask_params)

        optimize_fn = losses.optimization_manager(config)
        continuous = config.training.continuous
        likelihood_weighting = config.training.likelihood_weighting
        return losses.get_step_fn(self.sde, train=True, optimize_fn=optimize_fn,
                                  reduce_mean=config.training.reduce_mean, continuous=continuous,
                                  likelihood_weighting=likelihood_weighting,
                                  auxiliary_loss=config.training.auxiliary_loss,  # auxiliary loss
                                  random_mask=config.training.random_mask,  # ambient Diffusion, not used
                                  **kwargs)
    
    def setup_step_fn_val(self, config):
        """Create validation step function for reconstruction loss (train=False)."""
        kwargs = {}
        if config.training.auxiliary_loss:
            body_model_val = BodyModel(bm_path=self.bodymodel_path,
                                    num_betas=10,
                                    batch_size=config.eval.batch_size,
                                    model_type='smplx').to(self.device)
            for param in body_model_val.parameters():
                param.requires_grad = False
            aux_params = {'denormalize': self.denormalize_fn, 'body_model': body_model_val,
                        'model_type': "body", 'denoise_steps': config.training.denoise_steps}
            kwargs.update(aux_params)
        if config.training.random_mask:
            mask_params = {'min_mask_rate': config.training.min_mask_rate,
                        'max_mask_rate': config.training.max_mask_rate,
                        'observation_type': config.training.observation_type}
            kwargs.update(mask_params)

        # No optimize_fn for validation (train=False)
        continuous = config.training.continuous
        likelihood_weighting = config.training.likelihood_weighting
        
        return losses.get_step_fn(
            self.sde, 
            train=False,  # Critical: validation mode (no gradients)
            optimize_fn=None,
            reduce_mean=config.training.reduce_mean,
            continuous=continuous,
            likelihood_weighting=likelihood_weighting,
            auxiliary_loss=config.training.auxiliary_loss,
            random_mask=config.training.random_mask,
            **kwargs
        )

    def setup_sde(self, config):
        # Setup SDEs as per your configuration
        if config.training.sde.lower() == 'vpsde':
            return sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                                 N=config.model.num_scales)
        elif config.training.sde.lower() == 'subvpsde':
            return sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                                    N=config.model.num_scales)
        elif config.training.sde.lower() == 'vesde':
            return sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                                 N=config.model.num_scales)
        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    def training_step(self, batch, batch_idx):
        poses = self.normalize_fn(batch['body_pose'], from_axis=True)
        text_embeds = self.encode_text(batch['caption'])
        # Forward pass and calculate loss
        loss_dict = self.train_step_fn(self.model, batch=poses, condition=text_embeds, mask=None)

        # Log the losses (PyTorch Lightning respects log_every_n_steps)
        for key, value in loss_dict.items():
            self.log(f"{key}", value, prog_bar=True, logger=True)

        # Log to wandb only every log_freq steps (to reduce logging overhead)
        if self.global_step % self.config.training.log_freq == 0:
            loss_dict_scalar = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
            log_dict_to_wandb(loss_dict_scalar, self.global_step, prefix="train")

        return loss_dict['loss']

    def on_train_batch_end(self, *args):
        self.model_ema.update(self.model.parameters())

    def on_validation_epoch_start(self) -> None:
        # Store and copy EMA parameters for validation
        self.model_ema.store(self.model.parameters())
        self.model_ema.copy_to(self.model.parameters())
        self.compfn = DPoserComp(self.model, self.sde,
                                 self.config.training.continuous,)
    
    def validation_step(self, batch, batch_idx):
        poses = self.normalize_fn(batch['body_pose'], from_axis=True)
        text_list = batch['caption']  # Get raw text prompts (not embeddings)
        text_embeds_val = self.encode_text(text_list)
        
        # 🔴 Step 1: Compute reconstruction loss (recon_param_mse/recon_joint_l2)
        with torch.no_grad():  # Ensure no gradients (validation)
            loss_dict = self.val_step_fn(self.model, batch=poses, condition=text_embeds_val, mask=None)
        
        # Original validation logic (unchanged)
        eval_metrics, trajs, samples = self.process_validation_batch(poses, text_embeds_val)
        self.all_samples.append(samples)

        # --------------------------
        # Save text prompts for the last batch
        # --------------------------
        # Save trajs/text prompts for visualization (first validation batch only)
        max_val_batches = min(len(self.val_dataloader()), self.trainer.limit_val_batches)
        if batch_idx == 0 and batch_idx < max_val_batches:
            self.last_trajs = trajs if trajs is not None else torch.zeros((100, poses.shape[0], poses.shape[1]), device=poses.device)
            self.last_text_prompts = text_list[:5] if (trajs is not None and len(text_list)>=5) else []

        # --------------------------
        # 🔴 Step 2: Add all loss terms to eval_metrics (consistent naming with training)
        # All loss terms are now always present in loss_dict (set to 0.0 when not used)
        # --------------------------
        def to_scalar(v):
            return v.item() if isinstance(v, torch.Tensor) else v
        
        eval_metrics['loss'] = to_scalar(loss_dict.get('loss', torch.tensor(0.0)))
        eval_metrics['diffusion_loss'] = to_scalar(loss_dict.get('diffusion_loss', torch.tensor(0.0)))
        # Note: v2v_loss and j2j_loss not logged in validation - MPVPE and MPJPE are logged instead
        eval_metrics['recon_param_mse'] = to_scalar(loss_dict.get('recon_param_mse', torch.tensor(0.0)))
        eval_metrics['recon_joint_l2'] = to_scalar(loss_dict.get('recon_joint_l2', torch.tensor(0.0)))

        # --------------------------
        # logging
        # --------------------------
        log_dict_to_wandb(eval_metrics, self.global_step, prefix="val")
        for metric_name, metric_value in eval_metrics.items():
            self.log(
                f'val_{metric_name}', 
                metric_value, 
                sync_dist=True, 
                logger=True,
                batch_size=poses.shape[0],  # Critical for epoch averaging
                prog_bar=(metric_name in ['recon_param_mse', 'mpjpe'])  # Show reconstruction metrics in progress bar
            )
        
        return eval_metrics

    def on_validation_epoch_end(self) -> None:
        print(f"🔧 Config: training.render = {self.config.training.render}")
        print(f"🔧 last_trajs exists? {self.last_trajs is not None}")
        self.model_ema.restore(self.model.parameters())
        
        # 🔴 Step 1: Calculate epoch-averaged reconstruction metrics
        val_recon_param_mse = self.trainer.callback_metrics.get('val_recon_param_mse', torch.tensor(0.0))
        val_recon_joint_l2 = self.trainer.callback_metrics.get('val_recon_joint_l2', torch.tensor(0.0))
        
        # Log epoch-averaged values (clean curve)
        self.log(
            'val_recon_param_mse_epoch', 
            val_recon_param_mse, 
            sync_dist=True, 
            logger=True,
            prog_bar=True
        )
        self.log(
            'val_recon_joint_l2_epoch', 
            val_recon_joint_l2, 
            sync_dist=True, 
            logger=True
        )

        '''     ******* Compute APD and SI *******     '''
        all_results = torch.cat(self.all_samples, dim=0)[:250]
        body_pose = self.denormalize_fn(all_results, to_axis=True)
        body_out = self.body_model_eval(body_pose=body_pose)
        joints3d = body_out.Jtr
        body_joints3d = joints3d[:, :22, :]
        APD = average_pairwise_distance(body_joints3d)
        SI = self_intersections_percentage(body_out.v, body_out.f).mean()
        log_dict_to_wandb({'APD': APD.item(), 'SI': SI.item()}, self.global_step, prefix="val")
        self.log('APD', APD.item(), sync_dist=True, logger=True)
        self.log('SI', SI.item(), sync_dist=True, logger=True)

        if self.config.training.render and self.last_trajs is not None:
            # Use the stored trajs and all_results of the last batch
            self.render_and_log_images(self.last_trajs, all_results)

        # Reset the list of samples
        self.all_samples = []

    @torch.no_grad()
    def process_validation_batch(self, poses, text_embeds_val):
        eval_metrics = {'bpd': [], 'mpvpe': [], 'mpjpe': []}

        '''     ******* task1 bpd *******     '''
        bpd, z, nfe = self.likelihood_fn(self.model, poses, condition=text_embeds_val)
        eval_metrics['bpd'] = bpd.mean().item()

        '''     ******* task2 completion *******     '''
        mask, observation = create_mask(poses, part='left_leg', model='body')

        hypo_num = 10
        multihypo_denoise = []
        for hypo in range(hypo_num):
            completion = self.compfn.optimize(observation, mask, condition=text_embeds_val)
            multihypo_denoise.append(completion)
        multihypo_denoise = torch.stack(multihypo_denoise, dim=1)

        preds = self.denormalize_fn(multihypo_denoise, to_axis=True)
        gts = self.denormalize_fn(poses, to_axis=True)
        evaler = Evaler(body_model=self.body_model_vis, part='left_leg')
        eval_results = evaler.multi_eval_bodys(preds, gts)
        eval_metrics['mpvpe'] = eval_results['mpvpe'].mean().item()
        eval_metrics['mpjpe'] = eval_results['mpjpe'].mean().item()

        '''      ******* task3 generation *******     '''
        trajs, samples = self.sampling_fn(
            self.model,
            observation=None,
            condition=text_embeds_val,
        )  # [t, b, j*6], [b, j*6]

        return eval_metrics, trajs, samples

    
    def render_and_log_images(self, trajs, all_results):
        """Render and log images using training visualization utilities."""
        render_and_log_images(
            trajs, all_results,
            self.denormalize_fn, self.sde,
            self.body_model_vis, self.logger, self.global_step, self.current_epoch,
            text_prompts=self.last_text_prompts, pose_dim=self.POSE_DIM
        )


    def configure_optimizers(self):
        # Set up the optimizer
        optimizer = self.get_optimizer(self.config, self.model.parameters())

        # Set up the learning rate scheduler
        if self.config.optim.warmup > 0:
            lr_scheduler = {
                'scheduler': CosineWarmupScheduler(optimizer,
                                                   self.config.optim.warmup, self.config.training.n_iters),
                'interval': 'step',
            }
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def get_optimizer(self, config, params):
        if config.optim.optimizer == 'Adam':
            return optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999),
                              eps=config.optim.eps, weight_decay=config.optim.weight_decay)
        elif config.optim.optimizer == 'AdamW':
            return optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.98),
                               eps=config.optim.eps, weight_decay=config.optim.weight_decay)
        elif config.optim.optimizer == 'RAdam':
            return optim.RAdam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999),
                               eps=config.optim.eps, weight_decay=config.optim.weight_decay)
        else:
            raise NotImplementedError(f'Optimizer {config.optim.optimizer} not supported yet!')

    def on_save_checkpoint(self, checkpoint):
        checkpoint['model_ema'] = self.model_ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        self.model_ema.load_state_dict(checkpoint['model_ema'])
    
    @torch.no_grad()
    def generate_pose_from_text(self, text_prompt):
        text_embed = self.encode_text([text_prompt])
        
        trajs, samples = self.sampling_fn(
            self.model,
            observation=None,
            condition=text_embed
        )
        
        generated_pose = self.denormalize_fn(samples[0], to_axis=True)  # [21, 3]
        return generated_pose


def main(args, config, try_resume):
    pl.seed_everything(config.seed)
    config.name = args.name
    data_path = os.path.join(args.data_root, args.version, 'train')
    init_wandb(config, args)

    # Initialize the PyTorch Lightning data module and model
    data_module = AMASSDataModule(config, args)
    data_module.setup(stage='fit')
    val_loader = data_module.val_dataloader()
    print(f"Validation DataLoader size: {len(val_loader)} batches")
    if len(val_loader) == 0:
        raise ValueError("Validation DataLoader is empty! Check dataset path/version.")
    model = DPoserTrainer(config, args.bodymodel_path, data_path, N_POSES,
                          train_loader=data_module.train_dataloader(),
                          val_loader=val_loader,
                          pretrained_ckpt=args.pretrained_ckpt,
                          freeze_pretrained=args.freeze_pretrained)  
    # model = DPoserTrainer(config, args.bodymodel_path, data_path, N_POSES,
    #                       train_loader=data_module.train_dataloader(),
    #                       val_loader=data_module.val_dataloader(), )

    # Define logger and callbacks
    logger = TensorBoardLogger(f"logs/dposer/{config.dataset}", name=args.name)
    ckpt_dir = f"checkpoints/dposer/{config.dataset}/{args.name}"
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                          filename='{epoch:02d}-{step}-{val_mpjpe:.2f}',
                                          every_n_train_steps=config.training.save_freq,
                                          save_top_k=3, save_last=True, monitor='val_mpjpe', mode='min')
    model_logger = ModelSizeCallback()
    time_monitor = TimerCallback()
    lr_monitor = LearningRateMonitor()

    # Resume training
    resume_from_checkpoint = None
    if args.resume_ckpt is not None:
        resume_from_checkpoint = os.path.join(ckpt_dir, args.resume_ckpt)
        print('Resuming the training from {}'.format(resume_from_checkpoint))
    elif try_resume:
        available_ckpts = os.path.join(ckpt_dir, 'last.ckpt')
        if os.path.exists(available_ckpts):
            resume_from_checkpoint = os.path.realpath(available_ckpts)
            print('Resuming the training from {}'.format(resume_from_checkpoint))

    # Initialize the trainer
    trainer = pl.Trainer(
        accelerator='cuda',
        devices=config.devices,
        strategy='auto',  # 'ddp' is for multi-gpu training
        max_steps=config.training.n_iters,
        num_sanity_val_steps=5,
        val_check_interval=config.training.eval_freq,
        check_val_every_n_epoch=None,
        log_every_n_steps=config.training.log_freq,
        gradient_clip_val=config.optim.grad_clip,
        logger=logger,
        callbacks=[model_logger, time_monitor, lr_monitor, checkpoint_callback],
        benchmark=True,
        limit_val_batches=20  # 20,
    )

    # Train the model, lightning will move the model to GPU
    # Using `.to(self.device)` with `self.device` in `__init__` causes things to lock onto the CPU, and later when the entire module is moved to the GPU, the error "cuda:0 vs cpu" occurs.
    trainer.fit(model, ckpt_path=resume_from_checkpoint)

    wandb.finish()


if __name__ == '__main__':
    args = parse_args(sys.argv)
    config = import_configs(args.config_path)
    # FIXME: there seems to be a bug in PyTorch Lightning while loading EMA parameters from resume.
    resume_training_if_possible = False
    main(args, config, resume_training_if_possible)
