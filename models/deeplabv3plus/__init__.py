"""
models/deeplabv3plus
--------------------
DeepLabV3+ for multispectral remote sensing semantic segmentation.

Public API
----------
    deeplabv3plus        – full model
    DeepLabBackbone      – standalone backbone
    DeepLabV3PlusDecoder – standalone decoder
"""

from .deeplabv3plus import DeepLabV3Plus
from .backbone      import DeepLabBackbone
from .decoder       import DeepLabV3PlusDecoder

__all__ = [
    "DeepLabV3Plus",
    "DeepLabBackbone",
    "DeepLabV3PlusDecoder",
]