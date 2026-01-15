#!/usr/bin/env python
"""
Script to compute normalization parameters for body poses and global orientation.
This should be run once before training to create the normalization files.

{data_path}/{rot_rep}_normalize1.pt (if min_max=True)
{data_path}/{rot_rep}_normalize2.pt (if min_max=False)

Usage:
    # For z-score normalization (default, recommended):
    python scripts/compute_normalization_params.py --data-root ./data/body_data --version version1 --rot-rep axis
    
    # For min-max normalization:
    python scripts/compute_normalization_params.py --data-root ./data/body_data --version version1 --rot-rep axis --min-max True
"""

import os
import sys
import argparse
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.dataset.body import AMASSDataModule
from lib.dataset.utils import calculate_normalize_params
from lib.utils.generic import import_configs


def parse_args():
    parser = argparse.ArgumentParser(description='Compute normalization parameters for poses and global orientation')
    parser.add_argument('--config-path', '-c', type=str,
                        default='configs.body.subvp.timefc.get_config',
                        help='config files to build dataset')
    parser.add_argument('--data-root', type=str,
                        default='./data/body_data',
                        help='dataset root')
    parser.add_argument('--version', type=str, default='version1', help='dataset version')
    parser.add_argument('--rot-rep', type=str, default='axis', choices=['axis', 'rot6d'],
                        help='rotation representation')
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser.add_argument('--min-max', type=str2bool, default=False,
                        help='use min-max normalization (default: False, uses z-score/mean-std). Use --min-max True or omit for False')
    parser.add_argument('--batch-size', type=int, default=12800,
                        help='batch size for computing statistics')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='number of workers for DataLoader')
    parser.add_argument('--split-num', type=int, default=1,
                        help='number of splits for memory-efficient computation')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = import_configs(args.config_path)
    
    # Setup data module
    data_module = AMASSDataModule(config, args)
    data_module.setup(stage='fit')
    
    # Get training dataset
    train_dataset = data_module.train_dataloader().dataset
    
    # Determine output directory (where normalization files will be saved)
    # This MUST match where Posenormalizer expects to find them
    # Posenormalizer looks for files at: {data_path}/{rot_rep}_normalize1.pt or normalize2.pt
    # Where data_path = os.path.join(args.data_root, args.version, 'train')
    output_dir = os.path.join(args.data_root, args.version, 'train')
    
    print("=" * 80)
    print("Computing normalization parameters")
    print("=" * 80)
    print(f"Dataset: {len(train_dataset)} samples")
    print(f"Rotation representation: {args.rot_rep}")
    print(f"Normalization type: {'min-max' if args.min_max else 'z-score (mean/std)'}")
    print(f"Output directory: {output_dir}")
    print(f"(This matches where Posenormalizer expects to find the files)")
    print("=" * 80)
    
    # Compute normalization parameters
    # Note: The function now always computes global_orient statistics
    calculate_normalize_params(
        dataset=train_dataset,
        data_key='body_pose',  # Key for body pose in dataset batch
        rot_rep=args.rot_rep,
        min_max=args.min_max,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=output_dir,
        split_num=args.split_num
    )
    
    print("=" * 80)
    print("✅ Normalization parameters computed successfully!")
    print(f"Files saved to: {output_dir}")
    print(f"\nSaved files:")
    if args.min_max:
        normalize_file = os.path.join(output_dir, f'{args.rot_rep}_normalize1.pt')
        print(f"  - {normalize_file}")
    else:
        normalize_file = os.path.join(output_dir, f'{args.rot_rep}_normalize2.pt')
        print(f"  - {normalize_file}")
    
    # Verify files exist
    if os.path.exists(normalize_file):
        print(f"\n✅ File exists and is ready to use!")
        # Check if global_orient stats are included
        params = torch.load(normalize_file)
        if 'min_global_orient' in params or 'mean_global_orient' in params:
            print("✅ Global orientation statistics included")
        else:
            print("⚠️  Warning: Global orientation statistics not found in file")
    else:
        print(f"\n❌ Error: File not found at {normalize_file}")
    
    print("=" * 80)
    print(f"\n📝 Note: When training, make sure your training script uses:")
    print(f"   --data-root {args.data_root}")
    print(f"   --version {args.version}")
    print(f"   (This ensures Posenormalizer finds the files at the correct path)")
    print("=" * 80)


if __name__ == '__main__':
    main()
