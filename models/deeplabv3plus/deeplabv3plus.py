"""
DeepLabV3+ – Full Segmentation Model
--------------------------------------
Assembles:
  DeepLabBackbone  (backbone.py)          ← ResNet from torchvision, pretrained
      ↓  low (B, 256, H/4, W/4)
      ↓  high (B, 2048, H/16, W/16)
  ASPP             (torchvision)           ← reused directly
      ↓  (B, 256, H/16, W/16)
  DeepLabV3PlusDecoder  (decoder.py)       ← decoder+ (~50 lines custom)
      ↓
  logits (B, num_classes, H, W)

What comes from torchvision:
  - ResNet backbone with ImageNet weights
  - Full ASPP (ASPPConv, ASPPPooling, projection)

What is custom (~80 lines total):
  - Multispectral input adaptation (conv1 replaced)
  - Decoder+ with low-level skip connection

Notes
----
The ASPPPooling in torchvision has a BN on a 1×1 feature map that requires
batch_size > 1 during training. The fix applied here replaces that BN
with nn.Identity(), which is mathematically equivalent after convergence
(BN on a single value does not provide useful information).

Usage
-----
    from models.deeplabv3plus import deeplabv3plus

    model = deeplabv3plus(
        backbone    = "resnet101",
        in_channels = 12,
        num_classes = 14,
        pretrained_backbone = True,   # ImageNet (only if in_channels==3)
    )
    logits = model(x)   # x: (B, 12, H, W)  →  (B, 14, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import ASPP

from .backbone import DeepLabBackbone
from .decoder  import DeepLabV3PlusDecoder


def _fix_aspp_pooling_bn(aspp: ASPP) -> None:
    """
    Replaces the BatchNorm2d in the ASPPPooling branch with nn.Identity.
    Required to support batch_size=1 during training.
    """
    # The last module in aspp.convs is ASPPPooling (nn.Sequential)
    pool_branch = aspp.convs[-1]   # ASPPPooling is an nn.Sequential
    for i, layer in enumerate(pool_branch):
        if isinstance(layer, nn.BatchNorm2d):
            pool_branch[i] = nn.Identity()
            break


class DeepLabV3Plus(nn.Module):
    """
    Parameters
    ----------
    backbone    : "resnet50" | "resnet101"
    in_channels : int    input bands (default 3 RGB; use 12 for Sentinel-2)
    num_classes : int    segmentation classes
    pretrained_backbone : bool   ImageNet weights for backbone (only if in_channels==3)
    aspp_channels       : int    ASPP output channels (default 256)
    low_proj_ch         : int    low-level projection in the decoder (default 48)
    dropout             : float  dropout in the decoder
    """

    def __init__(
        self,
        backbone    : str   = "resnet101",
        in_channels : int   = 3,
        num_classes : int   = 14,
        pretrained_backbone : bool  = True,
        aspp_channels       : int   = 256,
        low_proj_ch         : int   = 48,
        dropout             : float = 0.1,
    ):
        super().__init__()

        # ── 1. Backbone ───────────────────────────────────────────────────
        self.backbone = DeepLabBackbone(
            name        = backbone,
            in_channels = in_channels,
            pretrained  = pretrained_backbone,
        )
        high_ch = self.backbone.channels["high"]   # 2048
        low_ch  = self.backbone.channels["low"]    # 256

        # ── 2. ASPP (from torchvision) ─────────────────────────────────────
        self.aspp = ASPP(high_ch, [12, 24, 36])
        _fix_aspp_pooling_bn(self.aspp)            # BN fix for batch=1

        # ── 3. Decoder+ ───────────────────────────────────────────────────
        self.decoder = DeepLabV3PlusDecoder(
            low_channels  = low_ch,
            aspp_channels = aspp_channels,
            num_classes   = num_classes,
            low_proj_ch   = low_proj_ch,
            dropout       = dropout,
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_channels, H, W)

        Returns
        -------
        logits : (B, num_classes, H, W)
        """
        features = self.backbone(x)
        aspp_out = self.aspp(features["high"])
        logits   = self.decoder(features["low"], aspp_out)
        return logits

    # ------------------------------------------------------------------
    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze the backbone."""
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def parameter_groups(
        self,
        lr_backbone : float = 1e-4,
        lr_head     : float = 1e-3,
        weight_decay: float = 4e-5,
    ) -> list:
        """Separate learning rates for backbone (fine-tuning) and ASPP+decoder (from scratch)."""
        head_params = list(self.aspp.parameters()) + list(self.decoder.parameters())
        return [
            {"params": self.backbone.parameters(), "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": head_params,                "lr": lr_head,     "weight_decay": weight_decay},
        ]

    def count_parameters(self) -> dict:
        def n(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "backbone": n(self.backbone),
            "aspp"    : n(self.aspp),
            "decoder" : n(self.decoder),
            "total"   : n(self),
        }