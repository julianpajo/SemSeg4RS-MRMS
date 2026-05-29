"""
SegFormer Decoder Heads
-----------------------

SegFormerHead
    Original SegFormer all-MLP decoder head. Fuses multi-scale features
    [F1, F2, F3, F4] via per-scale linear projections and produces
    full-resolution logits. Used when the BRD decoder is disabled.

ClassifierHead
    Lightweight classifier head that operates directly on F5, the
    boundary-refined feature map output by BRDDecoder (already at H/4).
    Pipeline: Conv3×3 → BN → GeLU → Dropout → Conv1×1 → 4× bilinear UP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerHead(nn.Module):
    """
    All-MLP decoder head from the original SegFormer architecture.

    Projects each encoder stage to a common embedding dimension, upsamples
    all scales to F1 resolution (H/4), concatenates them, and applies a
    fusion block followed by a classifier pipeline to produce full-resolution
    logits.

    Args:
        in_channels: Channel counts for [F1, F2, F3, F4],
                    e.g. [64, 128, 320, 512].
        embed_dim:   Unified projection dimension for all MLP layers (default 256).
        num_classes: Number of output segmentation classes (default 14).
        dropout:     Dropout probability applied before the final Conv1×1 (default 0.1).
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
        Args:
            features: List of encoder feature maps [F1, F2, F3, F4],
                    at spatial resolutions [H/4, H/8, H/16, H/32].

        Returns:
            Logits of shape (B, num_classes, H, W) at full input resolution,
            obtained via 4× bilinear upsampling.
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
# ClassifierHead  –  receives F5 from BRDDecoder  (B, C, H/4, W/4)
# ---------------------------------------------------------------------------

class ClassifierHead(nn.Module):
    """
    Lightweight classifier head for the BRD-enabled pipeline.

    Receives F5 from BRDDecoder (already at H/4) and produces full-resolution
    logits via a simple conv pipeline followed by 4× bilinear upsampling.

        F5 → Conv3×3 → BN → GeLU → Dropout → Conv1×1 → 4× UP → logits

    Args:
        in_channels: Number of channels in F5 (default 64).
        num_classes: Number of output segmentation classes (default 14).
        dropout:     Dropout probability applied before the final Conv1×1 (default 0.1).
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
        Args:
            f5: Boundary-refined feature map of shape (B, in_channels, H/4, W/4).

        Returns:
            Logits of shape (B, num_classes, H, W) at full input resolution.
        """
        x = self.conv3x3(f5)
        x = self.bn(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.conv1x1(x)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        return x