"""
Utilities for loading pre-trained checkpoints, especially when fine-tuning
from models without text conditioning to models with text conditioning.
"""

import torch
import numpy as np
from torch.serialization import add_safe_globals
from lib.algorithms.ema import ExponentialMovingAverage


def load_pretrained_checkpoint_for_finetuning(model, ckpt_path, device='cuda', 
                                             strict=False, is_ema=True,
                                             freeze_pretrained=False):
    """
    Load a pre-trained checkpoint that may not have text conditioning layers.
    
    This function is designed for fine-tuning scenarios where:
    - Pre-trained model: No text conditioning (original DPoser-X)
    - Target model: With text conditioning (your text-to-pose model)
    
    Args:
        model: The model to load weights into (with text conditioning)
        ckpt_path: Path to pre-trained checkpoint (.ckpt or .pth)
        device: Device to load checkpoint on
        strict: If False, allows missing keys (text conditioning layers)
        is_ema: Whether to load EMA weights and apply them
        freeze_pretrained: If True, freeze all non-text-conditioning parameters
    
    Returns:
        dict with loading statistics: {'loaded': int, 'missing': int, 'unexpected': int}
    """
    # Whitelist numpy scalars for PyTorch 2.6+
    add_safe_globals([np.core.multiarray.scalar])
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Extract model state dict
    if ckpt_path.endswith('.ckpt'):
        # PyTorch Lightning checkpoint
        state_dict = checkpoint.get('state_dict', {})
        # Remove 'model.' prefix if present
        model_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                model_state_dict[k[6:]] = v  # Remove 'model.' prefix
            elif not k.startswith('model_ema') and not k.startswith('optimizer') and not k.startswith('lr_schedulers'):
                # Direct model parameters (no prefix) - exclude non-model keys
                model_state_dict[k] = v
    else:
        # Standard PyTorch checkpoint
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        model_state_dict = state_dict if isinstance(state_dict, dict) else {}
    
    # Get current model state dict to identify text conditioning layers
    current_state_dict = model.state_dict()
    
    # Separate pretrained weights (exist in checkpoint) from new weights (text conditioning)
    pretrained_dict = {}
    missing_keys = []
    unexpected_keys = []
    
    for k, v in model_state_dict.items():
        if k in current_state_dict:
            # Check if shapes match
            if current_state_dict[k].shape == v.shape:
                pretrained_dict[k] = v
            else:
                print(f"⚠️  Shape mismatch for '{k}': checkpoint {v.shape} vs model {current_state_dict[k].shape} - skipping")
                missing_keys.append(k)
        else:
            unexpected_keys.append(k)
    
    # Find keys in current model that are not in checkpoint (likely text conditioning)
    for k in current_state_dict.keys():
        if k not in model_state_dict:
            missing_keys.append(k)
    
    # Load pretrained weights with strict=False to allow missing keys
    missing, unexpected = model.load_state_dict(pretrained_dict, strict=False)
    
    print(f"✅ Loaded {len(pretrained_dict)} parameters from checkpoint")
    if missing:
        print(f"📝 Missing keys (will use random init): {len(missing)}")
        for key in missing[:5]:  # Show first 5
            print(f"   - {key}")
        if len(missing) > 5:
            print(f"   ... and {len(missing) - 5} more")
    
    if unexpected:
        print(f"⚠️  Unexpected keys in checkpoint (ignored): {len(unexpected)}")
        for key in unexpected[:5]:
            print(f"   - {key}")
    
    # Handle EMA if present
    if is_ema and 'model_ema' in checkpoint:
        try:
            ema = ExponentialMovingAverage(model.parameters(), decay=checkpoint.get('model_ema', {}).get('decay', 0.999))
            ema.load_state_dict(checkpoint['model_ema'])
            ema.copy_to(model.parameters())
            print(f"✅ Applied EMA weights from checkpoint")
        except Exception as e:
            print(f"⚠️  Could not load EMA weights: {e}")
    
    # Freeze pretrained parameters if requested
    if freeze_pretrained:
        for name, param in model.named_parameters():
            # Only freeze if this parameter was loaded from checkpoint
            if name in pretrained_dict:
                param.requires_grad = False
                print(f"🔒 Frozen: {name}")
        print(f"🔒 Frozen {sum(1 for p in model.parameters() if not p.requires_grad)} pretrained parameters")
    
    return {
        'loaded': len(pretrained_dict),
        'missing': len(missing),
        'unexpected': len(unexpected_keys)
    }


def get_text_conditioning_parameters(model):
    """
    Get parameters related to text conditioning (for selective training).
    
    Returns:
        list of parameter names that are text conditioning related
    """
    text_params = []
    for name, _ in model.named_parameters():
        if 'text_pose_cross_attn' in name or 'text' in name.lower():
            text_params.append(name)
    return text_params


def set_requires_grad_for_text_conditioning_only(model, requires_grad=True):
    """
    Set requires_grad only for text conditioning parameters.
    Useful for two-stage fine-tuning:
    1. First train only text conditioning layers
    2. Then unfreeze everything for joint training
    """
    text_params = get_text_conditioning_parameters(model)
    
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Then, unfreeze text conditioning
    if requires_grad:
        for name, param in model.named_parameters():
            if name in text_params:
                param.requires_grad = True
        
        print(f"✅ Enabled gradients for {len(text_params)} text conditioning parameters:")
        for name in text_params:
            print(f"   - {name}")
    else:
        print(f"🔒 Frozen all parameters including text conditioning")
