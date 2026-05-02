"""
models/SegFormerSAE
-------------------
SegFormer + SAE + BRD per segmentazione semantica RS multispettrale.

Public API
----------
    SegFormerSAE    – modello completo
    SAEModule       – embedding spettrale standalone
    BRDDecoder      – Boundary-Refined Decoder standalone
    SegFormerHead   – All-MLP decoder head (modalità senza BRD)
    ClassifierHead  – classifier head leggero (modalità con BRD)
    MiT_CONFIGS     – configurazioni b0–b5
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
