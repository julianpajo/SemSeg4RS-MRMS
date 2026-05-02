"""
SegFormer Decoder Heads
-----------------------

SegFormerHead
    Decoder originale SegFormer: fonde [F1, F2, F3, F4] con proiezioni MLP.
    Usato quando BRD è disabilitato.

ClassifierHead
    Classifier head leggero che riceve F5 (output di BRDDecoder, già a H/4)
    e produce logits full-resolution.
    Pipeline: Conv3×3 → BN → GeLU → Dropout → Conv1×1 → 4× UP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerHead(nn.Module):
    """
    Parameters
    ----------
    in_channels : list[int]
        Channel sizes of [F1, F2, F3, F4], e.g. [64, 128, 320, 512].
    embed_dim : int
        Unified embedding dimension for the MLP layers (default 256).
    num_classes : int
        Number of segmentation classes.
    dropout : float
        Dropout probability before the final 1×1 conv.
    """

    def __init__(
        self,
        in_channels: list,
        embed_dim: int = 256,
        num_classes: int = 14,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert len(in_channels) == 4, "Expected 4-stage feature list"

        # Per-scale linear projections (implemented as 1×1 convs)
        self.linear_projections = nn.ModuleList([
            nn.Conv2d(c, embed_dim, kernel_size=1, bias=False)
            for c in in_channels
        ])

        # Fusion block  (matches the Classifier Head in Figure 1 of the paper)
        self.fusion_conv = nn.Conv2d(len(in_channels) * embed_dim, embed_dim, kernel_size=1, bias=False)

        # Classifier head: Conv3×3 → BN → GeLU → Dropout → Conv1×1
        self.conv3x3   = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False)
        self.bn        = nn.BatchNorm2d(embed_dim)
        self.gelu      = nn.GELU()
        self.dropout   = nn.Dropout2d(dropout)
        self.conv1x1   = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, features: list) -> torch.Tensor:
        """
        Parameters
        ----------
        features : [F1, F2, F3, F4]
            F1 is at H/4, F4 at H/32 (highest semantic level).

        Returns
        -------
        torch.Tensor  (B, num_classes, H, W)
            Logits at full input resolution (after 4× upsampling).
        """
        # Target spatial size = F1 resolution (H/4, W/4)
        target_h, target_w = features[0].shape[-2:]

        fused = []
        for feat, proj in zip(features, self.linear_projections):
            x = proj(feat)                                    # (B, embed_dim, Hi, Wi)
            x = F.interpolate(x, size=(target_h, target_w),  # upsample to H/4
                              mode="bilinear", align_corners=False)
            fused.append(x)

        # (B, 4*embed_dim, H/4, W/4)
        x = torch.cat(fused, dim=1)
        x = self.fusion_conv(x)          # (B, embed_dim, H/4, W/4)

        # Classifier head
        x = self.conv3x3(x)
        x = self.bn(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.conv1x1(x)              # (B, num_classes, H/4, W/4)

        # 4× bilinear upsampling → full resolution
        x = F.interpolate(x, scale_factor=4,
                          mode="bilinear", align_corners=False)
        return x


# ---------------------------------------------------------------------------
# ClassifierHead  –  riceve F5 da BRDDecoder  (B, C, H/4, W/4)
# ---------------------------------------------------------------------------

class ClassifierHead(nn.Module):
    """
    Classifier head leggero che opera su F5 (output di BRDDecoder).

    Pipeline: Conv3×3 → BN → GeLU → Dropout → Conv1×1 → 4× UP

    Parameters
    ----------
    in_channels : int   canali di F5 (default 64)
    num_classes : int   classi di segmentazione
    dropout     : float dropout prima del Conv1×1
    """

    def __init__(
        self,
        in_channels: int = 64,
        num_classes: int = 14,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.bn      = nn.BatchNorm2d(in_channels)
        self.gelu    = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)
        self.conv1x1 = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, f5: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        f5 : (B, in_channels, H/4, W/4)

        Returns
        -------
        logits : (B, num_classes, H, W)
        """
        x = self.conv3x3(f5)
        x = self.bn(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.conv1x1(x)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        return x
