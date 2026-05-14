"""
Spectral-Aware Embedding (SAE) Module
--------------------------------------
Implements the SAE module as described in the paper:

  X (B, 12, H, W)
    → BN per-band                          [Eq. 1]
    → Conv1×1 spectral projection (C1)     [Eq. 2]
    → Spectral Channel Attention (SE-like) [Eq. 3-4]
    → Spectral-Spatial Mixing (PW + DW)    [Eq. 5]
    → LayerNorm                            [Eq. 6]
    → Overlap Patch Embedding              (passed to SegFormer encoder)

References
----------
Figure 4 of the paper + Section 3.2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Spectral Channel Attention  (squeeze-and-excitation variant)
# ---------------------------------------------------------------------------

class SpectralChannelAttention(nn.Module):
    """
    GAP → Conv1×1 (C1 → C1//r) → ReLU → Conv1×1 (C1//r → C1) → Sigmoid
    Output is element-wise multiplied with the input feature map.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)          # (B, C1, 1, 1)
        self.fc1 = nn.Conv2d(channels, mid, 1)      # bottleneck
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid, channels, 1)      # restore channels
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C1, H, W)
        a = self.gap(x)               # (B, C1, 1, 1)
        a = self.fc1(a)               # (B, C1//r, 1, 1)
        a = self.relu(a)
        a = self.fc2(a)               # (B, C1, 1, 1)
        a = self.sigmoid(a)           # attention weights  [Eq. 3]
        return x * a                  # Fa = Fp ⊙ a        [Eq. 4]


# ---------------------------------------------------------------------------
# Spectral-Spatial Mixing  (pointwise + depthwise separable conv)
# ---------------------------------------------------------------------------

class SpectralSpatialMixing(nn.Module):
    """
    Conv1×1 (PW) → DWConv3×3 + residual with learnable α.

    Xmix = DWConv3×3(Conv1×1(Fa)) + α * Fp   [Eq. 5]
    """

    def __init__(self, channels: int, alpha_init: float = 0.1):
        super().__init__()
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.dw_conv = nn.Conv2d(
            channels, channels,
            kernel_size=3, padding=1,
            groups=channels,            # depthwise
            bias=False,
        )
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, fa: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
        # fa: attention-modulated features  (B, C1, H, W)
        # fp: projected features (residual) (B, C1, H, W)
        out = self.dw_conv(self.pw_conv(fa))   # (B, C1, H, W)
        return out + self.alpha * fp           # residual [Eq. 5]


# ---------------------------------------------------------------------------
# Full SAE Module
# ---------------------------------------------------------------------------

class SAEModule(nn.Module):
    """
    Spectral-Aware Embedding module.

    Parameters
    ----------
    in_channels : int
        Number of input spectral bands (default 12 for multispectral imagery).
    embed_dim : int
        Target embedding dimension C1 aligned with the first SegFormer stage
        (default 64, matching MiT-B2 / MiT-B5 stage-1).
    reduction : int
        Reduction ratio for the channel attention bottleneck (default 8).
    alpha_init : float
        Initial value for the learnable residual scaling factor α (default 0.1).
    """

    def __init__(
        self,
        in_channels: int = 12,
        embed_dim: int = 64,
        reduction: int = 8,
        alpha_init: float = 0.1,
    ):
        super().__init__()

        # --- Step 1: per-band Batch Normalization  [Eq. 1] ---
        self.band_norm = nn.BatchNorm2d(in_channels)

        # --- Step 2: spectral projection Conv1×1   [Eq. 2] ---
        self.spectral_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)

        # --- Step 3-4: Spectral Channel Attention  [Eq. 3-4] ---
        self.spectral_attn = SpectralChannelAttention(embed_dim, reduction=reduction)

        # --- Step 5: Spectral-Spatial Mixing       [Eq. 5] ---
        self.mixing = SpectralSpatialMixing(embed_dim, alpha_init=alpha_init)

        # --- Step 6: Layer Normalization           [Eq. 6] ---
        # Applied channel-last style via a reshape trick
        self.layer_norm = nn.LayerNorm(embed_dim)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  (B, in_channels, H, W)

        Returns
        -------
        torch.Tensor  (B, embed_dim, H, W)
            Spectrally-enhanced feature map ready for Overlap Patch Embedding.
        """
        # [Eq. 1]  per-band BN
        x1 = self.band_norm(x)

        # [Eq. 2]  spectral projection → Fp
        fp = self.spectral_proj(x1)          # (B, C1, H, W)

        # [Eq. 3-4]  channel attention → Fa
        fa = self.spectral_attn(fp)          # (B, C1, H, W)

        # [Eq. 5]  spectral-spatial mixing
        xmix = self.mixing(fa, fp)           # (B, C1, H, W)

        # [Eq. 6]  LayerNorm  (channel-last convention then back)
        B, C, H, W = xmix.shape
        xout = xmix.permute(0, 2, 3, 1)     # (B, H, W, C)
        xout = self.layer_norm(xout)
        xout = xout.permute(0, 3, 1, 2)     # (B, C, H, W)

        return xout