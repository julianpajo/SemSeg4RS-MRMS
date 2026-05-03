"""
DeepLabV3+ Decoder
-------------------
The only part not included in torchvision.
The original DeepLabV3+ decoder (Chen et al. 2018) is:

  low_level_features (H/4)
      → Conv1×1 (256 → 48)   [light projection]
      → Upsample ASPP output to H/4
      → Concat
      → Conv3×3 → BN → ReLU
      → Conv3×3 → BN → ReLU
      → Conv1×1 → logits
      → Upsample ×4 → full resolution

Reference: https://arxiv.org/abs/1802.02611  Figure 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabV3PlusDecoder(nn.Module):
    """
    Parameters
    ----------
    low_channels  : int   low-level feature channels (256 for ResNet)
    aspp_channels : int   ASPP output channels (256)
    num_classes   : int   number of segmentation classes
    low_proj_ch   : int   low-level projection channels (default 48, original paper)
    dropout       : float dropout before the final convolution
    """

    def __init__(
        self,
        low_channels  : int   = 256,
        aspp_channels : int   = 256,
        num_classes   : int   = 14,
        low_proj_ch   : int   = 48,
        dropout       : float = 0.1,
    ):
        super().__init__()

        # Low-level feature projection: 256 → 48
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_channels, low_proj_ch, 1, bias=False),
            nn.BatchNorm2d(low_proj_ch),
            nn.ReLU(inplace=True),
        )

        # Fusion and refinement
        fused_ch = aspp_channels + low_proj_ch
        self.refine = nn.Sequential(
            nn.Conv2d(fused_ch, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(
        self,
        low: torch.Tensor,
        aspp_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        low      : (B, low_channels,  H/4,  W/4)   low-level features
        aspp_out : (B, aspp_channels, H/16, W/16)   ASPP output

        Returns
        -------
        logits : (B, num_classes, H, W)
        """
        # Project low-level features
        low = self.low_proj(low)                              # (B, 48, H/4, W/4)

        # Upsample ASPP output to H/4 resolution
        aspp_up = F.interpolate(
            aspp_out,
            size=low.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )                                                     # (B, 256, H/4, W/4)

        # Concat + refine
        x = torch.cat([aspp_up, low], dim=1)                  # (B, 304, H/4, W/4)
        x = self.refine(x)                                    # (B, 256, H/4, W/4)

        # Classify
        x = self.classifier(x)                                # (B, C, H/4, W/4)

        # Upsample ×4 → full resolution
        x = F.interpolate(
            x,
            scale_factor=4,
            mode="bilinear",
            align_corners=False
        )
        return x