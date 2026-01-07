import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.algorithms.advanced.module import GaussianFourierProjection, get_sigmas, get_timestep_embedding, get_act


def create_model(model_config, N_POSES, POSE_DIM, text_embedding_dim=768):
    if 'FC' in model_config.type:
        model = TimeFC(
            model_config,
            n_poses=N_POSES,
            pose_dim=POSE_DIM,
            hidden_dim=model_config.HIDDEN_DIM,
            embed_dim=model_config.EMBED_DIM,
            n_blocks=model_config.N_BLOCKS,
            text_embedding_dim=text_embedding_dim,
        )
    elif model_config.type == 'TimeMLPs':
        model = TimeMLPs(
            model_config,
            n_poses=N_POSES,
            pose_dim=POSE_DIM,
            hidden_dim=model_config.HIDDEN_DIM,
            n_blocks=model_config.N_BLOCKS,
            text_embedding_dim=text_embedding_dim,
        )
    else:
        raise NotImplementedError('unsupported model')

    return model


class TextPoseCrossAttention(nn.Module):
    def __init__(self, pose_hidden_dim, text_embedding_dim=768, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = pose_hidden_dim // num_heads
        
        # Projection layers for cross-attention
        self.q_proj = nn.Linear(pose_hidden_dim, pose_hidden_dim)     # Query (pose features)
        self.k_proj = nn.Linear(text_embedding_dim, pose_hidden_dim)  # Key (text embeddings)
        self.v_proj = nn.Linear(text_embedding_dim, pose_hidden_dim)  # Value (text embeddings)
        self.out_proj = nn.Linear(pose_hidden_dim, pose_hidden_dim)   # Output projection
        
        # Layer normalization (stabilizes training)
        self.norm_pose = nn.LayerNorm(pose_hidden_dim)
        self.norm_text = nn.LayerNorm(text_embedding_dim)

    def forward(self, pose_feat, text_feat):
        # Layer normalization
        pose_feat_norm = self.norm_pose(pose_feat)
        text_feat_norm = self.norm_text(text_feat)
        
        # Reshape for multi-head attention
        B = pose_feat.shape[0]
        # Project to queries/keys/values and reshape for multi-head attention
        # Add a dummy sequence dimension (1) for valid attention (text/pose are [B, D] → [B, 1, D])
        q = self.q_proj(pose_feat_norm).reshape(B, 1, self.num_heads, self.head_dim).transpose(1,2)  # [B, num_heads, 1, head_dim]
        k = self.k_proj(text_feat_norm).reshape(B, 1, self.num_heads, self.head_dim).transpose(1,2)  # [B, num_heads, 1, head_dim]
        v = self.v_proj(text_feat_norm).reshape(B, 1, self.num_heads, self.head_dim).transpose(1,2)  # [B, num_heads, 1, head_dim]

        # Compute attention scores [B, num_heads, 1, 1]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=q.device))
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v).transpose(1,2).reshape(B, -1)  # [B, pose_hidden_dim]
        
        # Final projection + residual connection
        attn_output = self.out_proj(attn_output)
        pose_feat = pose_feat + attn_output  # Residual connection (critical for training stability)
        
        return pose_feat


class TimeMLPs(torch.nn.Module):
    def __init__(self, config, n_poses=21, pose_dim=6, hidden_dim=64, n_blocks=2, text_embedding_dim=768):
        super().__init__()
        dim = n_poses * pose_dim
        self.act = get_act(config)
        self.text_embedding_dim = text_embedding_dim

        layers = [torch.nn.Linear(dim + 1, hidden_dim),
                  self.act]

        for _ in range(n_blocks):
            layers.extend([
                torch.nn.Linear(hidden_dim, hidden_dim),
                self.act,
                torch.nn.Dropout(p=config.dropout)
            ])

        layers.append(torch.nn.Linear(hidden_dim, dim))

        self.net = torch.nn.Sequential(*layers)

        self.text_pose_cross_attn = TextPoseCrossAttention(
            pose_hidden_dim=hidden_dim,
            text_embedding_dim=text_embedding_dim,
            num_heads=8
        )

    def forward(self, batch, t, condition=None, mask=None):
        x = torch.cat([batch, t], dim=1)  # [B, dim + 1]
        x = self.net[:1](x)  # Initial linear + activation (output: [B, hidden_dim])
        if condition is not None:
            x = self.text_pose_cross_attn(x, condition)  # [B, hidden_dim]
        for layer in self.net[1:]:
            x = layer(x)
        return x


class TimeFC(nn.Module):
    """
    Independent condition feature projection layers for each block
    """

    def __init__(self, model_config, n_poses=21, pose_dim=6, hidden_dim=64,
                 embed_dim=32, n_blocks=2, text_embedding_dim=768):
        super(TimeFC, self).__init__()
        self.model_config = model_config
        self.n_poses = n_poses
        self.joint_dim = pose_dim
        self.n_blocks = n_blocks
        self.text_embedding_dim = text_embedding_dim

        self.act = get_act(model_config)

        self.pre_dense = nn.Linear(n_poses * pose_dim, hidden_dim)
        self.pre_dense_t = nn.Linear(embed_dim, hidden_dim)
        self.pre_gnorm = nn.GroupNorm(32, num_channels=hidden_dim)
        self.dropout = nn.Dropout(p=model_config.dropout)

        # time embedding
        self.time_embedding_type = model_config.embedding_type.lower()
        if self.time_embedding_type == 'fourier':
            self.gauss_proj = GaussianFourierProjection(embed_dim=embed_dim, scale=model_config.fourier_scale)
        elif self.time_embedding_type == 'positional':
            self.posit_proj = functools.partial(get_timestep_embedding, embedding_dim=embed_dim)
        else:
            assert 0

        self.shared_time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            self.act,
        )
        self.register_buffer('sigmas', torch.tensor(get_sigmas(model_config), dtype=torch.float))

        for idx in range(n_blocks):
            setattr(self, f'b{idx + 1}_dense1', nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f'b{idx + 1}_dense1_t', nn.Linear(embed_dim, hidden_dim))
            setattr(self, f'b{idx + 1}_gnorm1', nn.GroupNorm(32, num_channels=hidden_dim))

            setattr(self, f'b{idx + 1}_dense2', nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f'b{idx + 1}_dense2_t', nn.Linear(embed_dim, hidden_dim))
            setattr(self, f'b{idx + 1}_gnorm2', nn.GroupNorm(32, num_channels=hidden_dim))

        self.post_dense = nn.Linear(hidden_dim, n_poses * pose_dim)

        self.text_pose_cross_attn = TextPoseCrossAttention(
            pose_hidden_dim=hidden_dim,
            text_embedding_dim=text_embedding_dim,
            num_heads=8  
        )

    def forward(self, batch, t, condition=None, mask=None):
        """
        batch: [B, j*3] or [B, j*6]
        t: [B]
        condition: [B, 768] CLIP text embeddings
        mask: [B, j*3] or [B, j*6] same dim as batch
        Return: [B, j*3] or [B, j*6] same dim as batch
        """
        bs = batch.shape[0]
        # Make sure ALL inputs are on the same device as the model
        device = next(self.parameters()).device
        batch = batch.to(device)
        t = t.to(device)
        if condition is not None:
            condition = condition.to(device)
        if mask is not None:
            mask = mask.to(device)

        # time embedding
        if self.time_embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = t  
            temb = self.gauss_proj(torch.log(used_sigmas))
        elif self.time_embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = t
            self.sigmas = self.sigmas
            used_sigmas = self.sigmas[t.long()]
            temb = self.posit_proj(timesteps)  # get_timestep_embedding
        else:
            raise ValueError(f'time embedding type {self.time_embedding_type} unknown.')

        # Critical: some functions may return CPU tensors, so force to GPU
        temb = temb.to(device)
        temb = self.shared_time_embed(temb)

        h = self.pre_dense(batch)
        h += self.pre_dense_t(temb)
        h = self.pre_gnorm(h)
        h = self.act(h)
        h = self.dropout(h)
    
        if condition is not None:
            h = self.text_pose_cross_attn(h, condition)

        for idx in range(self.n_blocks):
            h1 = getattr(self, f'b{idx + 1}_dense1')(h)
            h1 += getattr(self, f'b{idx + 1}_dense1_t')(temb)
            h1 = getattr(self, f'b{idx + 1}_gnorm1')(h1)
            h1 = self.act(h1)
            # dropout, maybe
            h1 = self.dropout(h1)

            h2 = getattr(self, f'b{idx + 1}_dense2')(h1)
            h2 += getattr(self, f'b{idx + 1}_dense2_t')(temb)
            h2 = getattr(self, f'b{idx + 1}_gnorm2')(h2)
            h2 = self.act(h2)
            # dropout, maybe
            h2 = self.dropout(h2)

            h = h + h2

        res = self.post_dense(h)  # [B, j*3]

        ''' normalize the output '''
        if self.model_config.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((bs, 1))
            res = res / used_sigmas

        return res