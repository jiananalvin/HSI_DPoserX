import os
import sys
# Add DPoser-X root to Python path (fixes "No module named 'lib'")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import json
from functools import partial
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from absl import flags, app
from absl.flags import argparse_flags
from ml_collections.config_flags import config_flags

# Import your core modules (now works with PYTHONPATH fix)
from lib.algorithms.advanced import likelihood, sde_lib, sampling
from lib.algorithms.advanced.model import create_model
from lib.body_model.body_model import BodyModel
from lib.body_model.visual import render_mesh, multiple_render
from lib.dataset.body.AMASS import AMASSDataset
from lib.utils.generic import load_model
from lib.dataset.utils import Posenormalizer

# OpenCLIP for text encoding
import open_clip

# --------------------------
# Fix Broken AMASSDataModule
# --------------------------
N_POSES = 21


class AMASSDataModule(pl.LightningDataModule):
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.args = args

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = AMASSDataset(
                root_path=self.args.data_root,
                subset='train',
                posescript_data_dir=self.config.data.posescript_data_dir,
                amass_dir=self.config.data.amass_dir,
                use_human_annotations=self.config.data.use_human_annotations
            )
            self.sample_weights = torch.ones(len(self.train_dataset))

            self.val_dataset = AMASSDataset(
                root_path=self.args.data_root,
                version=self.args.version,
                subset='val',
                posescript_data_dir=self.config.data.posescript_data_dir,
                amass_dir=self.config.data.amass_dir,
                use_human_annotations=self.config.data.use_human_annotations
            )

        if stage == 'test' or stage is None:
            self.test_dataset = AMASSDataset(
                root_path=self.args.data_root,
                subset='test',
                posescript_data_dir=self.config.data.posescript_data_dir,
                amass_dir=self.config.data.amass_dir,
                use_human_annotations=self.config.data.use_human_annotations
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            num_workers=8,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.eval.batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.eval.batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=True
        )


# --------------------------
# Configure Flags
# --------------------------
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Inference configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])

# Render settings
bg_img = np.ones([512, 384, 3]) * 255  # background canvas
focal = [1500, 1500]
princpt = [200, 192]


# --------------------------
# Parse Args
# --------------------------
def parse_args(argv):
    parser = argparse_flags.ArgumentParser(description='Text-conditioned pose generation on test set')

    parser.add_argument('--data-root', type=str, default='./data/body_data')
    parser.add_argument('--version', type=str, default='version1')
    parser.add_argument('--posescript-data-dir', type=str,
                        default="/home/jxudt/HSI_GenPose/data/posescript_release")
    parser.add_argument('--amass-dir', type=str,
                        default="/home/jxudt/HSI_GenPose/data/amass")
    parser.add_argument('--use-human-annotations', action='store_true', default=True)

    parser.add_argument('--bodymodel-path', type=str,
                        default='./body_models/smplx/SMPLX_NEUTRAL.npz')
    parser.add_argument('--ckpt-path', type=str,
                        default="./checkpoints/dposer/amass/reproduce_body/last-v8.ckpt")

    parser.add_argument('--num-samples-per-caption', type=int, default=5)
    parser.add_argument('--max-test-samples', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--output-path', type=str, default='./output/body/test_inference')

    # keep this arg for compatibility, but we will IGNORE it and force faster=False
    parser.add_argument('--faster-render', action='store_true', default=True)

    parser.add_argument('--view', type=str, default='front')

    args = parser.parse_args(argv[1:])
    return args


# --------------------------
# Text-Conditioned Inferencer
# --------------------------
class TextConditionedInferencer:
    def __init__(self, args, config):
        self.args = args
        self.config = config

        # Inference can be on CUDA
        self.device = torch.device(args.device)

        # Rendering MUST be CPU (and we will avoid PyTorch3D path)
        self.render_device = torch.device("cpu")

        # Override config with PoseScript paths
        self.config.data.posescript_data_dir = args.posescript_data_dir
        self.config.data.amass_dir = args.amass_dir
        self.config.data.use_human_annotations = args.use_human_annotations

        # SDE
        self.sde = self._setup_sde()
        self.sampling_eps = 1e-3 if self.config.training.sde.lower() in ['vpsde', 'subvpsde'] else 1e-5

        # Model
        self.pose_dim = 3 if config.data.rot_rep == 'axis' else 6
        self.model = self._setup_model()

        # Normalizer
        self.normalizer = Posenormalizer(
            data_path=os.path.join(args.data_root, args.version, 'train'),
            normalize=config.data.normalize,
            min_max=config.data.min_max,
            rot_rep=config.data.rot_rep,
            device=self.device
        )
        self.denormalize_fn = self.normalizer.offline_denormalize

        # CLIP
        self.clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self.clip_model = self.clip_model.to(self.device)
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # Body model FOR RENDERING ONLY -> keep on CPU
        self.body_model = BodyModel(
            bm_path=args.bodymodel_path,
            num_betas=10,
            batch_size=args.num_samples_per_caption,
            model_type='smplx'
        ).to(self.render_device)

        # Sampler
        self.sampling_fn = self._setup_sampling_fn()

        # ------------------------------------------------------------
        # CRITICAL FIX:
        # Force faster=False so we DO NOT go into visual.py's PyTorch3D
        # faster_render() path that causes GPU/CPU mismatch.
        # ------------------------------------------------------------
        self.save_renders = partial(
            multiple_render,
            bg_img=bg_img,
            focal=focal,
            princpt=princpt,
            device=self.render_device,  # cpu
            view=args.view,
            faster=False,               # <<< IMPORTANT: disable PyTorch3D path
            part='body',
            convert=False
        )

    def _setup_sde(self):
        sde_name = self.config.training.sde.lower()
        if sde_name == 'vpsde':
            return sde_lib.VPSDE(beta_min=self.config.model.beta_min,
                                 beta_max=self.config.model.beta_max,
                                 N=self.config.model.num_scales)
        elif sde_name == 'subvpsde':
            return sde_lib.subVPSDE(beta_min=self.config.model.beta_min,
                                    beta_max=self.config.model.beta_max,
                                    N=self.config.model.num_scales)
        elif sde_name == 'vesde':
            return sde_lib.VESDE(sigma_min=self.config.model.sigma_min,
                                 sigma_max=self.config.model.sigma_max,
                                 N=self.config.model.num_scales)
        else:
            raise NotImplementedError(f"SDE {self.config.training.sde} not supported")

    def _setup_model(self):
        model = create_model(self.config.model, N_POSES, self.pose_dim)
        model = model.to(self.device)
        model.eval()
        load_model(model, self.config.model, self.args.ckpt_path, self.device, is_ema=True)
        print(f"✅ Loaded model from checkpoint: {self.args.ckpt_path}")
        return model

    def _setup_sampling_fn(self):
        sampling_shape = (self.args.num_samples_per_caption, N_POSES * self.pose_dim)
        inverse_scaler = lambda x: x
        return sampling.get_sampling_fn(
            self.config,
            self.sde,
            sampling_shape,
            inverse_scaler,
            self.sampling_eps,
            device=self.device
        )

    @torch.no_grad()
    def encode_text(self, text_list):
        tokens = self.clip_tokenizer([t.strip().lower() for t in text_list]).to(self.device)
        embeds = self.clip_model.encode_text(tokens)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        return embeds

    @torch.no_grad()
    def generate_poses(self, text_embed):
        text_embed = text_embed.repeat(self.args.num_samples_per_caption, 1)
        _, samples = self.sampling_fn(self.model, observation=None, condition=text_embed)
        samples_denorm = self.denormalize_fn(samples, to_axis=True)
        return samples_denorm

    def save_results(self, sample_idx, pose_id, gt_pose, generated_poses, caption):
        sample_folder = Path(self.args.output_path) / f"sample_{sample_idx:04d}"
        sample_folder.mkdir(parents=True, exist_ok=True)

        # Render inputs on CPU
        gt_pose_cpu = gt_pose.detach().cpu()
        generated_poses_cpu = generated_poses.detach().cpu()
        body_model_cpu = self.body_model  # already cpu

        # Render GT
        self.save_renders(
            samples=gt_pose_cpu.unsqueeze(0),
            denormalize_fn=None,
            model=body_model_cpu,
            target_path=str(sample_folder),
            img_name='gt.png'
        )

        # Render generated samples
        for i, pose in enumerate(generated_poses_cpu):
            self.save_renders(
                samples=pose.unsqueeze(0),
                denormalize_fn=None,
                model=body_model_cpu,
                target_path=str(sample_folder),
                img_name=f"sample_{i}.png"
            )

        # Save PoseID and caption
        with open(sample_folder / "pose_id.txt", "w") as f:
            f.write(str(pose_id))
        with open(sample_folder / "caption.txt", "w") as f:
            f.write(caption)

        metadata = {
            "sample_idx": int(sample_idx),
            "pose_id": str(pose_id),
            "caption": caption,
            "num_generated_samples": int(len(generated_poses_cpu)),
            "checkpoint_path": self.args.ckpt_path,
            "use_human_annotations": bool(self.args.use_human_annotations),
            "view": self.args.view,
            "inference_device": str(self.device),
            "render_device": str(self.render_device),
            "render_mode": "multiple_render(faster=False)"
        }
        with open(sample_folder / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"✅ Saved results for sample {sample_idx:04d} (PoseID: {pose_id}) to {sample_folder}")


# --------------------------
# Main
# --------------------------
def main(args):
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_grad_enabled(False)

    config = FLAGS.config
    inferencer = TextConditionedInferencer(args, config)

    print("\n📥 Loading PoseScript test set...")

    data_module = AMASSDataModule(
        config=config,
        args=SimpleNamespace(data_root=args.data_root, version=args.version)
    )
    data_module.setup(stage='test')
    test_loader = data_module.test_dataloader()
    print(f"✅ Loaded PoseScript test set with {len(test_loader)} batches")

    total_samples = 0
    for batch_idx, batch in enumerate(test_loader):
        if args.max_test_samples and total_samples >= args.max_test_samples:
            break

        gt_body_poses = batch['body_pose'].to(inferencer.device)  # inference device
        captions = batch['caption']
        pose_ids = batch['pose_id']

        for sample_idx_in_batch, (gt_pose, pose_id, caption) in enumerate(zip(gt_body_poses, pose_ids, captions)):
            global_sample_idx = total_samples + sample_idx_in_batch
            if args.max_test_samples and global_sample_idx >= args.max_test_samples:
                break

            text_embed = inferencer.encode_text([caption])
            print(f"\nGenerating {args.num_samples_per_caption} samples for sample {global_sample_idx:04d} (PoseID: {pose_id})...")
            generated_poses = inferencer.generate_poses(text_embed)

            inferencer.save_results(
                sample_idx=global_sample_idx,
                pose_id=pose_id,
                gt_pose=gt_pose,
                generated_poses=generated_poses,
                caption=caption
            )

        total_samples += len(gt_body_poses)

    print("\n🎉 Inference complete!")
    print(f"📊 Processed {total_samples} PoseScript test samples")
    print(f"📁 Results saved to: {args.output_path}")


if __name__ == '__main__':
    app.run(main, flags_parser=parse_args)