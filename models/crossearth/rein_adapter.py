"""
Rein – Reins adapter for DINOv2
---------------------------------
Standalone implementation of Rein (CVPR 2024):
  "Stronger, Fewer, & Superior: Harnessing Vision Foundation Models
   for Domain Generalized Semantic Segmentation"
  https://arxiv.org/abs/2312.04265

Rein inserts learnable tokens into the hidden state of each ViT block
through a shared low-rank MLP, without modifying the backbone weights:

  f'_i = f_i + Rein(f_i)

The trainable parameters are:
  - learnable_tokens:  (num_layers, num_tokens, token_dim), i.e. ~0.5M params
  - MLP shared across all layers
  - scale, a learnable per-layer scalar

The DINOv2 backbone remains fully frozen.

Reference: https://github.com/w1oves/Rein (adapted for standalone use)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReinAdapter(nn.Module):
    """
    Rein adapter for ViT backbones.

    Parameters
    ----------
    num_layers  : int   number of transformer blocks in the backbone
                        (12 for ViT-B, 24 for ViT-L, 32 for ViT-H)
    embed_dim   : int   backbone hidden-state dimension
                        (768 ViT-B, 1024 ViT-L, 1280 ViT-H)
    num_tokens  : int   number of learnable tokens per layer (default 100)
    token_dim   : int   internal token dimension (default 256)
    """

    def __init__(
        self,
        num_layers : int,
        embed_dim  : int,
        num_tokens : int = 100,
        token_dim  : int = 256,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim  = embed_dim
        self.num_tokens = num_tokens

        # Learnable tokens: one set for each layer
        self.learnable_tokens = nn.Parameter(
            torch.empty(num_layers, num_tokens, token_dim)
        )
        nn.init.trunc_normal_(self.learnable_tokens, std=0.02)

        # MLP shared across all layers, projecting tokens → residual
        # input:  embed_dim  (image features)
        # output: embed_dim  (residual to add)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + token_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # Learnable per-layer scale, initialized at 0 → no initial perturbation
        self.scale = nn.Parameter(torch.zeros(num_layers))

    # ------------------------------------------------------------------
    def forward_layer(
        self,
        features: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Computes the Rein residual for a single layer.

        Parameters
        ----------
        features  : (B, N, D)  ViT hidden state after block `layer_idx`
        layer_idx : int

        Returns
        -------
        features' : (B, N, D)  refined features
        """
        B, N, D = features.shape
        tokens = self.learnable_tokens[layer_idx]      # (num_tokens, token_dim)

        # Expand tokens across the batch and sequence
        # Use max pooling over features as global context
        ctx = features.max(dim=1).values              # (B, D)  global context
        ctx = ctx.unsqueeze(1).expand(B, self.num_tokens, D)  # (B, T, D)

        tok = tokens.unsqueeze(0).expand(B, -1, -1)   # (B, T, token_dim)
        x   = torch.cat([ctx, tok], dim=-1)            # (B, T, D+token_dim)
        res = self.mlp(x)                              # (B, T, D)

        # Aggregate the residual with mean pooling over tokens → (B, D)
        # and broadcast it across the whole sequence
        res = res.mean(dim=1, keepdim=True)            # (B, 1, D)

        scale = torch.tanh(self.scale[layer_idx])      # learnable scale ∈ (-1,1)
        return features + scale * res

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)