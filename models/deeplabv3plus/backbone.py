"""
Backbone for DeepLabV3+
-----------------------
Uses ResNet from torchvision with IntermediateLayerGetter to extract:
  - low_level  (layer1): (B, 256,  H/4,  W/4)   → decoder skip connection
  - high_level (layer4): (B, 2048, H/16, W/16)  → ASPP input

Default output stride is 16 (layer3 has dilation=2 in torchvision ResNet).

Available backbones:
  "resnet50"  | "resnet101"
"""

import torch
import torch.nn as nn
from torchvision.models import (
    resnet50,  ResNet50_Weights,
    resnet101, ResNet101_Weights,
)
from torchvision.models._utils import IntermediateLayerGetter


# Low-level and high-level channels for each backbone
BACKBONE_CHANNELS = {
    "resnet50" : {"low": 256, "high": 2048},
    "resnet101": {"low": 256, "high": 2048},
}

_FACTORIES = {
    "resnet50" : (resnet50,  ResNet50_Weights.IMAGENET1K_V1),
    "resnet101": (resnet101, ResNet101_Weights.IMAGENET1K_V1),
}


class DeepLabBackbone(nn.Module):
    """
    ResNet backbone that returns low-level and high-level features.

    Parameters
    ----------
    name        : "resnet50" | "resnet101"
    in_channels : int   number of input bands (default 3 RGB).
                        If != 3, the first conv layer is replaced.
    pretrained  : bool  load ImageNet weights (only if in_channels == 3)
    """

    def __init__(
        self,
        name        : str  = "resnet101",
        in_channels : int  = 3,
        pretrained  : bool = True,
    ):
        super().__init__()

        factory, weights = _FACTORIES[name]
        w = weights if (pretrained and in_channels == 3) else None
        base = factory(weights=w)

        # Adapt the first conv layer for multispectral input
        if in_channels != 3:
            old = base.conv1
            base.conv1 = nn.Conv2d(
                in_channels, old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False,
            )
            if pretrained:
                # Initialize by averaging RGB weights across new channels
                with torch.no_grad():
                    base.conv1.weight.copy_(
                        old.weight.mean(dim=1, keepdim=True).expand_as(base.conv1.weight)
                        / in_channels * 3
                    )

        self.body = IntermediateLayerGetter(
            base,
            return_layers={"layer1": "low", "layer4": "high"},
        )
        self.channels = BACKBONE_CHANNELS[name]

    def forward(self, x: torch.Tensor) -> dict:
        """
        Returns
        -------
        dict with keys "low" and "high"
          low  : (B, 256,  H/4,  W/4)
          high : (B, 2048, H/16, W/16)
        """
        return self.body(x)