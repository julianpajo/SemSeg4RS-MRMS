"""
ReinAdapter – Reins PEFT adapter for frozen ViT backbones
-----------------------------------------------------------
Standalone PyTorch implementation of Rein (CVPR 2024):
  "Stronger, Fewer, & Superior: Harnessing Vision Foundation Models
   for Domain Generalized Semantic Segmentation"
  https://arxiv.org/abs/2312.04265

Rein injects a lightweight residual into each ViT block's hidden state
via a shared low-rank MLP and per-layer learnable tokens, leaving all
backbone weights unchanged:

    f'_i = f_i + scale_i · Rein(f_i)

Trainable parameters (~0.5M total for ViT-L with defaults):
    learnable_tokens:  (num_layers, num_tokens, token_dim)
    shared MLP:        (embed_dim + token_dim) → embed_dim
    scale:             (num_layers,)  — one learnable scalar per layer

Reference: https://github.com/w1oves/Rein
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReinAdapter(nn.Module):
    """
    Rein adapter for frozen ViT backbones.

    Args:
        num_layers: Number of transformer blocks in the backbone
                    (e.g. 12 for ViT-B, 24 for ViT-L, 40 for ViT-G).
        embed_dim:  Backbone hidden-state dimension
                    (e.g. 768 for ViT-B, 1024 for ViT-L).
        num_tokens: Number of learnable tokens per layer (default 100).
        token_dim:  Internal token projection dimension (default 256).
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
        Apply the Rein residual to a single ViT block's output.

        A global context vector is derived from the hidden state via max pooling,
        concatenated with the layer's learnable tokens, and passed through the
        shared MLP. The resulting residual is averaged over tokens and added to
        all sequence positions, scaled by a learnable tanh-bounded scalar.

        Args:
            features:  Hidden state of shape (B, N, D) output by block layer_idx.
            layer_idx: Index of the current transformer block.

        Returns:
            Refined hidden state of shape (B, N, D).
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
        """
        Return the total number of trainable parameters in the adapter.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)