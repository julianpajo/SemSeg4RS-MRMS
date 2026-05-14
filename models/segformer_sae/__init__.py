"""
models/segformer_sae
-------------------
SegFormer + SAE + BRD for multispectral remote sensing semantic segmentation.

Public API
----------
    segformer_sae    – full model
    SAEModule       – standalone spectral embedding
    BRDDecoder      – standalone Boundary-Refined Decoder
    SegFormerHead   – All-MLP decoder head (mode without BRD)
    ClassifierHead  – lightweight classifier head (mode with BRD)
    MiT_CONFIGS     – b0–b5 configurations
"""

from .segformer_sae     import SegFormerSAE, MiT_CONFIGS
from .sae_module        import SAEModule
from .brd_decoder       import BRDDecoder
from .segformer_decoder import SegFormerHead, ClassifierHead

__all__ = [
    "SegFormerSAE",
    "SAEModule",
    "BRDDecoder",
    "SegFormerHead",
    "ClassifierHead",
    "MiT_CONFIGS",
]