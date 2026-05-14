"""
dofa Backbone – wrapper around torchgeo
----------------------------------------
Imports dofa directly from torchgeo (dofa_base_patch16_224 /
dofa_large_patch16_224) and adds forward hooks on the transformer
blocks to extract multi-scale features for segmentation.

dofa is a plain ViT with wave-dynamic DOFAEmbedding instead of the
standard patch embedding — it accepts images with any number of
bands as long as wavelengths are provided in µm.

Available variants:
  "small"  → dofa_small_patch16_224   embed=384,  depth=12  (no official weights)
  "base"   → dofa_base_patch16_224    embed=768,  depth=12  ← DOFABase16_Weights
  "large"  → dofa_large_patch16_224   embed=1024, depth=24  ← DOFALarge16_Weights
  "huge"   → dofa_huge_patch14_224    embed=1280, depth=32  (no official weights)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgeo.models import (
    dofa_small_patch16_224,
    dofa_base_patch16_224,
    dofa_large_patch16_224,
    #dofa_huge_patch14_224,
    DOFABase16_Weights,
    DOFALarge16_Weights,
)


DOFA_CONFIGS = {
    "small": dict(embed_dim=384,  depth=12, patch_size=16),
    "base" : dict(embed_dim=768,  depth=12, patch_size=16),
    "large": dict(embed_dim=1024, depth=24, patch_size=16),
    "huge" : dict(embed_dim=1280, depth=32, patch_size=14),
}

_OUT_INDICES = {
    12: [2,  5,  8,  11],
    24: [5,  11, 17, 23],
    32: [7,  15, 23, 31],
}

_FACTORIES = {
    "small": dofa_small_patch16_224,
    "base" : dofa_base_patch16_224,
    "large": dofa_large_patch16_224,
    #"huge" : dofa_huge_patch14_224,
}

_WEIGHTS = {
    "base" : DOFABase16_Weights.DOFA_MAE,
    "large": DOFALarge16_Weights.DOFA_MAE,
}


class DOFABackbone(nn.Module):
    """
    dofa backbone with multi-scale extraction via forward hooks.

    Parameters
    ----------
    variant     : "small" | "base" | "large" | "huge"
    pretrained  : bool   load pretrained MAE weights (base and large only)
    out_indices : list   block indices from which features are extracted
    """

    def __init__(
        self,
        variant    : str  = "base",
        pretrained : bool = True,
        out_indices: list = None,
    ):
        super().__init__()
        cfg = DOFA_CONFIGS[variant]
        self.embed_dim  = cfg["embed_dim"]
        self.patch_size = cfg["patch_size"]
        depth = cfg["depth"]

        self.out_indices = out_indices or _OUT_INDICES[depth]

        factory = _FACTORIES[variant]
        weights = _WEIGHTS.get(variant) if pretrained else None
        if pretrained and variant not in _WEIGHTS:
            print(f"[DOFABackbone] No pretrained weights for '{variant}'. Random init.")
            weights = None

        self.model = factory(weights=weights)
        # Do NOT modify global_pool: patch tokens are captured
        # by hooks on the blocks, BEFORE the final norm.

        self._feat_cache: dict = {}
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for idx in self.out_indices:
            self._hooks.append(
                self.model.blocks[idx].register_forward_hook(self._make_hook(idx))
            )

    def _make_hook(self, idx: int):
        def hook(module, input, output):
            self._feat_cache[idx] = output
        return hook

    def _interpolate_pos_embed(self, x: torch.Tensor):
        """Interpolates pos_embed for inputs with a size different from 224×224."""
        img_h, img_w = x.shape[-2], x.shape[-1]
        h = img_h // self.patch_size
        w = img_w // self.patch_size
        num_patches = h * w

        pos = self.model.pos_embed                    # (1, N_train+1, D)
        if pos.shape[1] - 1 == num_patches:
            return                                    # already correct

        cls_pos   = pos[:, :1, :]                     # (1, 1, D)
        patch_pos = pos[:, 1:, :]                     # (1, N_train, D)
        n_train   = patch_pos.shape[1]
        ht = wt   = int(math.sqrt(n_train))

        patch_pos = patch_pos.reshape(1, ht, wt, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(h, w),
                                  mode="bilinear", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)

        self.model.pos_embed = nn.Parameter(
            torch.cat([cls_pos, patch_pos], dim=1), requires_grad=False
        )

    def forward(self, x: torch.Tensor, wavelengths: list) -> list:
        """
        Parameters
        ----------
        x           : (B, C, H, W)
        wavelengths : list[float]   wavelengths in µm, len == C

        Returns
        -------
        list of 4× (B, embed_dim, H/p, W/p)
        """
        self._feat_cache.clear()
        h = x.shape[-2] // self.patch_size
        w = x.shape[-1] // self.patch_size

        self._interpolate_pos_embed(x)
        self.model.forward_features(x, wavelengths)

        features = []
        for idx in sorted(self._feat_cache.keys()):
            tokens = self._feat_cache[idx][:, 1:, :]     # remove cls
            B, N, D = tokens.shape
            features.append(tokens.transpose(1, 2).reshape(B, D, h, w))

        self._feat_cache.clear()
        return features

    def freeze(self, freeze: bool = True):
        for p in self.model.parameters():
            p.requires_grad = not freeze