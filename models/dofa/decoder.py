"""
Decoders for DOFASeg
---------------------
Since DOFA is a plain ViT, all four extracted feature maps share the same
spatial resolution (H/patch_size, W/patch_size) and channel dimension
(embed_dim). Two decoder variants are provided:

    LinearDecoder  — concatenates all scales, then applies a minimal
                     Conv1×1 projection pipeline. Lightweight baseline.

    MLADecoder     — Multi-Level Aggregation: projects each scale
                     independently, then accumulates features via a
                     progressive sum from the deepest to the shallowest
                     stage, followed by a Conv3×3 refinement block.
                     More expressive; recommended default.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearDecoder(nn.Module):
    """
    Minimal linear decoder for ViT-based backbones with isotropic feature maps.

    Concatenates all four feature maps along the channel dimension, then
    applies a two-stage Conv1×1 projection pipeline to produce logits, which
    are upsampled to full resolution by a single bilinear interpolation.

        Concat([F1, F2, F3, F4]) → Conv1×1 → BN → GeLU → Dropout → Conv1×1 → patch_size× UP

    Args:
        embed_dim:   Channel dimension of each input feature map
                    (e.g. 768 for base, 1024 for large).
        decoder_dim: Intermediate projection dimension (default 256).
        num_classes: Number of output segmentation classes (default 14).
        patch_size:  Backbone patch size used as the final upsampling factor
                    (16 for base/large, 14 for huge).
        dropout:     Dropout probability before the final Conv1×1 (default 0.1).
    """

    def __init__(
        self,
        embed_dim   : int   = 768,
        decoder_dim : int   = 256,
        num_classes : int   = 14,
        patch_size  : int   = 16,
        dropout     : float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.head = nn.Sequential(
            nn.Conv2d(embed_dim * 4, decoder_dim, 1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )

    def forward(self, features: list) -> torch.Tensor:
        x = torch.cat(features, dim=1)                         # (B, D*4, h, w)
        x = self.head(x)                                       # (B, C, h, w)
        return F.interpolate(x, scale_factor=self.patch_size,
                             mode="bilinear", align_corners=False)


class MLADecoder(nn.Module):
    """
    Multi-Level Aggregation (MLA) decoder for ViT-based backbones.

    Projects each of the four feature maps independently to a common
    decoder_dim, then aggregates them via a progressive element-wise sum
    from the deepest stage to the shallowest (F4 → F3 → F2 → F1).
    A Conv3×3 refinement block is applied to the aggregated features before
    upsampling to full resolution.

        Fi → Conv1×1 → BN → ReLU  (×4, independently)
        x  = F4_proj + F3_proj + F2_proj + F1_proj  (progressive accumulation)
        x  → Conv3×3 → BN → GeLU → Dropout → Conv1×1 → patch_size× UP

    Args:
        embed_dim:   Channel dimension of each input feature map
                    (e.g. 768 for base, 1024 for large).
        decoder_dim: Intermediate channel dimension after per-scale projection
                    (default 256 for base, 512 for large).
        num_classes: Number of output segmentation classes (default 14).
        patch_size:  Backbone patch size used as the final upsampling factor
                    (16 for base/large, 14 for huge).
        dropout:     Dropout probability before the final Conv1×1 (default 0.1).
    """

    def __init__(
        self,
        embed_dim   : int   = 768,
        decoder_dim : int   = 256,
        num_classes : int   = 14,
        patch_size  : int   = 16,
        dropout     : float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size

        # Per-scale projection embed_dim → decoder_dim
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
        Args:
            features: List of four feature maps [F1, F2, F3, F4],
                    each of shape (B, embed_dim, h, w), where h = H/patch_size.

        Returns:
            Logits of shape (B, num_classes, H, W) at full input resolution.
        """
        projs = [p(f) for p, f in zip(self.projs, features)]   # 4×(B, D, h, w)

        # Progressive sum from the deepest feature (F4 → F1)
        x = projs[-1]
        for p in reversed(projs[:-1]):
            x = x + p

        x = self.refine(x)                                      # (B, C, h, w)
        return F.interpolate(x, scale_factor=self.patch_size,
                             mode="bilinear", align_corners=False)