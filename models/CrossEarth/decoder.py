"""
Decoder for CrossEarth
-----------------------
DINOv2 produces 4 feature maps at the same resolution (H/14, W/14),
all with embed_dim channels. It is not a hierarchical encoder like MiT.

Two available decoders:

  LinearDecoder  (default, lightweight)
    Fuses the 4 feature maps with a linear projection and produces
    full-resolution logits. Used for baselines or limited resources.

  MLADecoder  (Multi-Level Aggregation, recommended for CrossEarth)
    Projects each scale → decoder_dim, sums them, refines with Conv3×3,
    then upsamples. More faithful to the multi-scale spirit of the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearDecoder(nn.Module):
    """
    Minimal linear decoder for plain ViT (4 scales at the same resolution).

    Pipeline:
      [F1, F2, F3, F4] → channel concat → Conv1×1 → BN → GeLU
      → Dropout → Conv1×1 (num_classes) → bilinear UP ×patch_size

    Parameters
    ----------
    embed_dim   : int   channels of each feature map (e.g. 1024 for ViT-L)
    num_scales  : int   number of input scales (default 4)
    decoder_dim : int   intermediate channels (default 512)
    num_classes : int   segmentation classes
    patch_size  : int   backbone patch size (14 for DINOv2)
    dropout     : float
    """

    def __init__(
        self,
        embed_dim   : int   = 1024,
        num_scales  : int   = 4,
        decoder_dim : int   = 512,
        num_classes : int   = 14,
        patch_size  : int   = 14,
        dropout     : float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        in_ch = embed_dim * num_scales

        self.head = nn.Sequential(
            nn.Conv2d(in_ch, decoder_dim, 1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )

    def forward(self, features: list) -> torch.Tensor:
        # All features are already at the same resolution (H/14, W/14)
        x = torch.cat(features, dim=1)                          # (B, D*4, h, w)
        x = self.head(x)                                        # (B, C, h, w)
        x = F.interpolate(x, scale_factor=self.patch_size,
                          mode="bilinear", align_corners=False)
        return x


class MLADecoder(nn.Module):
    """
    Multi-Level Aggregation decoder.

    Each scale is projected → decoder_dim, the 4 projected maps are summed
    progressively (from deepest to shallowest), then refined with Conv3×3
    and upsampled.

    More faithful to the architecture used in CrossEarth/Rein with DINOv2.

    Parameters
    ----------
    embed_dim   : int   channels of each feature map (e.g. 1024 for ViT-L)
    decoder_dim : int   intermediate channels (default 512)
    num_classes : int   segmentation classes
    patch_size  : int   backbone patch size (14 for DINOv2)
    dropout     : float
    """

    def __init__(
        self,
        embed_dim   : int   = 1024,
        decoder_dim : int   = 512,
        num_classes : int   = 14,
        patch_size  : int   = 14,
        dropout     : float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size

        # Per-scale projection
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, decoder_dim, 1, bias=False),
                nn.BatchNorm2d(decoder_dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(4)
        ])

        # Refinement after aggregation
        self.refine = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )

    def forward(self, features: list) -> torch.Tensor:
        """
        Parameters
        ----------
        features : [F1, F2, F3, F4]  — (B, embed_dim, h, w) each

        Returns
        -------
        logits : (B, num_classes, H, W)
        """
        # Project each scale
        projs = [proj(f) for proj, f in zip(self.projs, features)]

        # Progressive aggregation (sum from the deepest feature)
        x = projs[-1]
        for p in reversed(projs[:-1]):
            x = x + p

        x = self.refine(x)                                      # (B, C, h, w)
        x = F.interpolate(x, scale_factor=self.patch_size,
                          mode="bilinear", align_corners=False)
        return x