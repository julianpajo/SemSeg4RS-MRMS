"""
DINOv2 Backbone + Rein
-----------------------
Wraps the DINOv2 backbone, loaded from torch.hub, and injects
the ReinAdapter after each transformer block through forward hooks.

The DINOv2 backbone is fully frozen.
Only the ReinAdapter is trainable.

Returns multi-scale features reshaped to 2D, compatible with
any segmentation decoder: Linear, UperNet, Mask2Former-style.

DINOv2 variants available on torch.hub:
    "dinov2_vits14"  →  ViT-S/14  embed=384,  layers=12
    "dinov2_vitb14"  →  ViT-B/14  embed=768,  layers=12
    "dinov2_vitl14"  →  ViT-L/14  embed=1024, layers=24  ← CrossEarth uses this
    "dinov2_vitg14"  →  ViT-G/14  embed=1536, layers=40

Versions with registers, better for dense prediction:
    "dinov2_vits14_reg", "dinov2_vitb14_reg",
    "dinov2_vitl14_reg", "dinov2_vitg14_reg"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .rein_adapter import ReinAdapter


DINOV2_CONFIGS = {
    "dinov2_vits14"    : dict(embed_dim=384,  num_layers=12, patch_size=14),
    "dinov2_vitb14"    : dict(embed_dim=768,  num_layers=12, patch_size=14),
    "dinov2_vitl14"    : dict(embed_dim=1024, num_layers=24, patch_size=14),
    "dinov2_vitg14"    : dict(embed_dim=1536, num_layers=40, patch_size=14),
    "dinov2_vits14_reg": dict(embed_dim=384,  num_layers=12, patch_size=14),
    "dinov2_vitb14_reg": dict(embed_dim=768,  num_layers=12, patch_size=14),
    "dinov2_vitl14_reg": dict(embed_dim=1024, num_layers=24, patch_size=14),
    "dinov2_vitg14_reg": dict(embed_dim=1536, num_layers=40, patch_size=14),
}

# Layer indices from which multi-scale features are extracted
# CrossEarth uses the last 4 layers, as in Mask2Former
_MULTISCALE_LAYERS = {
    12: [2, 5, 8, 11],     # ViT-S/B: 4 uniformly spaced checkpoints
    24: [5, 11, 17, 23],   # ViT-L
    40: [9, 19, 29, 39],   # ViT-G
}


class DINOv2WithRein(nn.Module):
    """
    Frozen DINOv2 backbone + Rein PEFT adapter.

    Parameters
    ----------
    variant     : str    torch.hub model name, e.g. "dinov2_vitl14_reg"
    num_tokens  : int    learnable tokens per layer in the Rein adapter
    token_dim   : int    internal dimension of Rein tokens
    out_indices : list   layer indices from which features are extracted
                         (None = use the default for the variant)
    """

    def __init__(
        self,
        variant    : str  = "dinov2_vitl14_reg",
        num_tokens : int  = 100,
        token_dim  : int  = 256,
        out_indices: list = None,
    ):
        super().__init__()
        cfg = DINOV2_CONFIGS[variant]
        self.embed_dim  = cfg["embed_dim"]
        self.num_layers = cfg["num_layers"]
        self.patch_size = cfg["patch_size"]
        self._variant   = variant

        # Backbone placeholder — populated by load_backbone()
        self.backbone = None

        # Rein adapter (trainable)
        self.rein = ReinAdapter(
            num_layers = self.num_layers,
            embed_dim  = self.embed_dim,
            num_tokens = num_tokens,
            token_dim  = token_dim,
        )

        # Layers from which multi-scale features are extracted
        self.out_indices = out_indices or _MULTISCALE_LAYERS[self.num_layers]

        # Storage for intermediate features, populated by hooks
        self._features: list = []
        self._hooks: list = []

    # ------------------------------------------------------------------
    def load_backbone(self, pretrained: bool = True, force_reload: bool = False):
        """
        Loads DINOv2 from torch.hub (facebookresearch/dinov2).
        Requires an internet connection on the first call.
        """
        print(f"[CrossEarth] Loading backbone '{self._variant}' from torch.hub...")
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            self._variant,
            pretrained   = pretrained,
            force_reload = force_reload,
        )
        self.backbone.eval()
        # Freeze the entire backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._register_hooks()
        print(f"[CrossEarth] Backbone loaded and frozen.")
        return self

    def load_backbone_from_checkpoint(self, ckpt_path: str):
        """
        Loads DINOv2 from a local checkpoint
        downloaded from https://dl.fbaipublicfiles.com/dinov2/.

        Example
        -------
            backbone.load_backbone_from_checkpoint(
                "checkpoints/dinov2_vitl14_pretrain.pth"
            )
        """
        print(f"[CrossEarth] Loading backbone from '{ckpt_path}'...")
        # Creates the architecture without weights
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            self._variant,
            pretrained = False,
        )
        state = torch.load(ckpt_path, map_location="cpu")
        # Some checkpoints have keys under the "model" prefix
        if "model" in state:
            state = state["model"]
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        if missing:
            print(f"[CrossEarth] Missing weights ({len(missing)}): {missing[:3]} ...")
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._register_hooks()
        print(f"[CrossEarth] Backbone loaded from checkpoint.")
        return self

    # ------------------------------------------------------------------
    def _register_hooks(self):
        """Registers forward hooks on the blocks at out_indices."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        for idx in self.out_indices:
            # DINOv2 uses self.backbone.blocks[idx]
            hook = self.backbone.blocks[idx].register_forward_hook(
                self._make_hook(idx)
            )
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            # output: (B, N+1, D), where N+1 = cls + patch tokens
            # Applies Rein to the layer, only on patch tokens, not cls
            out = self.rein.forward_layer(output, layer_idx)
            self._features.append((layer_idx, out))
        return hook

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> list:
        """
        Parameters
        ----------
        x : (B, 3, H, W)   ImageNet-normalized image

        Returns
        -------
        features : list of (B, embed_dim, H/p, W/p)
            4 multi-scale feature maps from out_indices.
        """
        assert self.backbone is not None, \
            "Backbone not loaded. Call load_backbone() before forward."

        self._features.clear()
        img_h, img_w = x.shape[-2:]
        h = img_h // self.patch_size
        w = img_w // self.patch_size

        # Backbone forward pass; hooks capture intermediate features
        with torch.no_grad():
            _ = self.backbone(x)

        # Sort by layer index and reshape to 2D
        self._features.sort(key=lambda t: t[0])
        out = []
        for _, feat in self._features:
            # feat: (B, N+1, D) — remove cls token
            patch_tokens = feat[:, 1:, :]              # (B, N, D)
            B, N, D = patch_tokens.shape
            # reshape → (B, D, h, w)
            feat_2d = patch_tokens.transpose(1, 2).reshape(B, D, h, w)
            out.append(feat_2d)

        self._features.clear()
        return out   # [F1, F2, F3, F4] — all at (B, embed_dim, H/14, W/14)