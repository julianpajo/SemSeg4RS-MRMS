"""
Decoder for dofa Segmentation
--------------------------------
dofa is a plain ViT → all 4 feature maps come out at the same
spatial resolution (H/patch_size, W/patch_size) with embed_dim channels.

Two decoders:

  LinearDecoder  — concat + Conv1×1, minimal
  MLADecoder     — Multi-Level Aggregation, progressive sum + Conv3×3
                   more expressive, recommended by default
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearDecoder(nn.Module):
    """
    Linear decoder: concat 4 scales → Conv1×1 → BN → GeLU → Dropout → Conv1×1 → UP.

    Parameters
    ----------
    embed_dim   : int   channels of each feature map, e.g. 768 for base, 1024 for large
    decoder_dim : int   intermediate channels
    num_classes : int   segmentation classes
    patch_size  : int   backbone patch size (16 or 14) — final upsampling scale
    dropout     : float
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
    Multi-Level Aggregation decoder.

    Each scale → projection → decoder_dim, then progressive sum
    from the deepest feature, followed by Conv3×3 refinement + UP.

    Parameters
    ----------
    embed_dim   : int   channels of each feature map
    decoder_dim : int   intermediate channels (default 256 for base, 512 for large)
    num_classes : int   segmentation classes
    patch_size  : int   backbone patch size
    dropout     : float
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
        Parameters
        ----------
        features : [F1, F2, F3, F4]  — (B, embed_dim, h, w) each

        Returns
        -------
        logits : (B, num_classes, H, W)
        """
        projs = [p(f) for p, f in zip(self.projs, features)]   # 4×(B, D, h, w)

        # Progressive sum from the deepest feature (F4 → F1)
        x = projs[-1]
        for p in reversed(projs[:-1]):
            x = x + p

        x = self.refine(x)                                      # (B, C, h, w)
        return F.interpolate(x, scale_factor=self.patch_size,
                             mode="bilinear", align_corners=False)