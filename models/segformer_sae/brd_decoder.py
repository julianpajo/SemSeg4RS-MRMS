"""
Boundary-Refined Decoder (BRD)
-------------------------------
Generalized version of the WBRD (Wetland Boundary-Refined Decoder) from the paper,
renamed for use on generic remote sensing semantic segmentation datasets.

Architecture (Figures 5, 6, 7 of the paper):

  Encoder features: [F1, F2, F3, F4]
    F4 (512, H/32) → A → ┐
                          Concat → B → x
    F3 (320, H/16) ───────┘
         x        → A → ┐
                          Concat → B → x
    F2 (128, H/8)  ───────┘
         x        → A → ┐
                          Concat → B → F5
    F1  (64, H/4)  ───────┘

  A = DPR → MBA → 2× UP
  B = Conv3×3 → BN → GeLU

  Output F5: (B, 64, H/4, W/4)

Submodules:
  SobelFilter       – non-trainable gradient prior (∇F)
  DPRModule         – Dual-Path Refine  [Eq. 7]
  MBAModule         – Multi-scale Boundary Attention [Eq. 8-9]
  BRDDecoder        – progressive decoder [Eq. 10]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sobel Filter  (non-trainable gradient prior)
# ---------------------------------------------------------------------------

class SobelFilter(nn.Module):
    """
    Applies the Sobel filter independently to each channel (depthwise).
    It has no trainable parameters — it acts as a stable geometric prior.
    Output: gradient magnitude, same shape as the input.
    """

    def __init__(self):
        super().__init__()
        # Sobel Gx and Gy kernels
        kx = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1,-2,-1],
                            [ 0, 0, 0],
                            [ 1, 2, 1]], dtype=torch.float32)
        # shape: (1, 1, 3, 3) — expanded to (C, 1, 3, 3) in forward
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Expand kernels for each channel (depthwise)
        kx = self.kx.expand(C, 1, 3, 3)
        ky = self.ky.expand(C, 1, 3, 3)
        gx = F.conv2d(x, kx, padding=1, groups=C)
        gy = F.conv2d(x, ky, padding=1, groups=C)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)   # (B, C, H, W)


# ---------------------------------------------------------------------------
# DPR – Dual-Path Refine Module  [Eq. 7, Figure 6]
# ---------------------------------------------------------------------------

class DPRModule(nn.Module):
    """
    Path 1: DWConv3×3 → PWConv1×1 → BN+GeLU
    Path 2: Sobel(∇F) → Conv3×3   → BN+GeLU
    Concat → Conv1×1 → BN+GeLU → Output (same C as the input)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.sobel = SobelFilter()

        # Path 1 – texture (depthwise separable)
        self.dw_conv = nn.Conv2d(channels, channels, 3, padding=1,
                                 groups=channels, bias=False)
        self.pw_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn1     = nn.BatchNorm2d(channels)

        # Path 2 – boundary gradient
        self.edge_conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2       = nn.BatchNorm2d(channels)

        # Fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Path 1
        p1 = self.gelu(self.bn1(self.pw_conv(self.dw_conv(x))))

        # Path 2  (non-trainable Sobel)
        with torch.no_grad():
            grad = self.sobel(x)
        p2 = self.gelu(self.bn2(self.edge_conv(grad)))

        # Concat + fuse  →  Fdpr  [Eq. 7]
        return self.fuse(torch.cat([p1, p2], dim=1))


# ---------------------------------------------------------------------------
# MBA – Multi-scale Boundary Attention  [Eq. 8-9, Figure 7]
# ---------------------------------------------------------------------------

class MBAModule(nn.Module):
    """
    4 parallel branches: Conv3×3, Conv5×5, Conv7×7 + Sobel
    Concat → Conv1×1 → BN → Sigmoid → attention map A
    Output: F ⊙ A  (boundary-amplified features)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.sobel = SobelFilter()

        def branch(k):
            return nn.Sequential(
                nn.Conv2d(channels, channels, k, padding=k // 2, bias=False),
                nn.BatchNorm2d(channels),
                nn.GELU(),
            )

        self.b3 = branch(3)
        self.b5 = branch(5)
        self.b7 = branch(7)

        # Sobel branch
        self.b_edge = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        # Generates the attention map  [Eq. 8]
        self.attn = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m3 = self.b3(x)
        m5 = self.b5(x)
        m7 = self.b7(x)

        with torch.no_grad():
            grad = self.sobel(x)
        me = self.b_edge(grad)

        # A = σ(Conv1×1(Concat[M3, M5, M7, ∇F]))  [Eq. 8]
        A = self.attn(torch.cat([m3, m5, m7, me], dim=1))

        # Fmba = F ⊙ A  [Eq. 9]
        return x * A


# ---------------------------------------------------------------------------
# Block A  =  DPR → MBA → 2× Upsample
# ---------------------------------------------------------------------------

class BlockA(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dpr = DPRModule(channels)
        self.mba = MBAModule(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dpr(x)
        x = self.mba(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return x


# ---------------------------------------------------------------------------
# Block B  =  Conv3×3 → BN → GeLU   (after skip concat)
# ---------------------------------------------------------------------------

class BlockB(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# BRD Decoder  (Boundary-Refined Decoder)
# ---------------------------------------------------------------------------

class BRDDecoder(nn.Module):
    """
    Progressive upsampling decoder with skip connections and boundary-aware
    refinement.

    Pipeline [Eq. 10]:
      F4 → A4 → Concat(A4_up, F3) → B3 → x3
      x3 → A3 → Concat(A3_up, F2) → B2 → x2
      x2 → A2 → Concat(A2_up, F1) → B1 → F5

    Parameters
    ----------
    encoder_channels : list[int]
        Channels for [F1, F2, F3, F4], default [64, 128, 320, 512].
    out_channels : int
        F5 output channels (default 64, aligned with F1).
    """

    def __init__(
        self,
        encoder_channels: list = None,
        out_channels: int = 64,
    ):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [64, 128, 320, 512]

        C1, C2, C3, C4 = encoder_channels

        # Initial projection F4 → C3 (to align channels before Concat)
        self.proj4 = nn.Conv2d(C4, C3, 1, bias=False)

        # Step 1: F4 → A → 2×UP → Concat F3 → B  (output: C2)
        self.a4 = BlockA(C3)
        self.b3 = BlockB(C3 + C3, C2)   # Concat: A4_up(C3) + F3(C3)

        # F2 → C2 projection if needed (F2 is already C2=128, but kept for uniformity)
        # Step 2: x3 → A → 2×UP → Concat F2 → B  (output: C1)
        self.a3 = BlockA(C2)
        self.b2 = BlockB(C2 + C2, C1)   # Concat: A3_up(C2) + F2(C2)

        # Step 3: x2 → A → 2×UP → Concat F1 → B  (output: out_channels)
        self.a2 = BlockA(C1)
        self.b1 = BlockB(C1 + C1, out_channels)  # Concat: A2_up(C1) + F1(C1)

    def forward(self, features: list) -> torch.Tensor:
        """
        Parameters
        ----------
        features : [F1, F2, F3, F4]

        Returns
        -------
        F5 : (B, out_channels, H/4, W/4)
        """
        F1, F2, F3, F4 = features

        # Step 1
        x = self.proj4(F4)                          # (B, C3, H/32, W/32)
        x = self.a4(x)                              # (B, C3, H/16, W/16)
        x = self.b3(torch.cat([x, F3], dim=1))      # (B, C2, H/16, W/16)

        # Step 2
        x = self.a3(x)                              # (B, C2, H/8,  W/8)
        x = self.b2(torch.cat([x, F2], dim=1))      # (B, C1, H/8,  W/8)

        # Step 3
        x = self.a2(x)                              # (B, C1, H/4,  W/4)
        F5 = self.b1(torch.cat([x, F1], dim=1))     # (B, out, H/4, W/4)

        return F5