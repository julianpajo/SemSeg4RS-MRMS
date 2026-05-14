"""
models/dofa
-----------
dofa per segmentazione semantica RS multispettrale.

Paper:  "Neural Plasticity-Inspired Foundation Model for Observing
         the Earth Crossing Modalities"  — https://arxiv.org/abs/2403.15356

Architettura:
  dofa backbone (torchgeo, opzionalmente frozen)
      ↓  4 feature maps  (B, embed_dim, H/p, W/p)
  MLADecoder / LinearDecoder  (trainabile)
      ↓
  logits  (B, num_classes, H, W)

Cosa viene da torchgeo (zero codice custom):
  - dofa_base_patch16_224  / dofa_large_patch16_224
  - DOFABase16_Weights.DOFA_MAE  / DOFALarge16_Weights.DOFA_MAE
  - DOFAEmbedding (wave-dynamic patch embed)

Cosa è scritto custom (~80 righe totali):
  - Forward hooks per estrarre features intermedie  (dofa_backbone.py)
  - MLADecoder / LinearDecoder  (decoder.py)

Dipendenze:
  pip install torchgeo

Usage
-----
    from models.dofa import DOFASeg, S2_WAVELENGTHS

    # Backbone frozen, fine-tune solo decoder
    model = DOFASeg(variant="base", num_classes=14, pretrained=True, freeze_backbone=True)
    logits = model(x, S2_WAVELENGTHS)   # x: (B, 12, H, W)

    # Backbone unfrozen, fine-tune tutto
    model = DOFASeg(variant="large", num_classes=14, pretrained=True, freeze_backbone=False)

Varianti disponibili (con pesi pretrained):
  "base"   → dofa_base_patch16_224   (embed=768,  ~86M params)
  "large"  → dofa_large_patch16_224  (embed=1024, ~307M params)

Varianti senza pesi ufficiali (random init):
  "small"  → dofa_small_patch16_224  (embed=384)
  "huge"   → dofa_huge_patch14_224   (embed=1280)

Public API
----------
    DOFASeg        – modello completo
    DOFABackbone   – backbone standalone (con hooks)
    MLADecoder     – decoder MLA standalone
    LinearDecoder  – decoder lineare standalone
    DOFA_CONFIGS   – configurazioni varianti
    S2_WAVELENGTHS – wavelengths standard Sentinel-2 (12 bande, µm)
"""

from .dofa_seg      import DOFASeg
from .dofa_backbone import DOFABackbone, DOFA_CONFIGS
from .decoder       import MLADecoder, LinearDecoder

# Wavelengths standard Sentinel-2 (12 bande, µm)
# Ordine: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12
S2_WAVELENGTHS = [
    0.443,   # B1  - Coastal aerosol
    0.490,   # B2  - Blue
    0.560,   # B3  - Green
    0.665,   # B4  - Red
    0.705,   # B5  - Red Edge 1
    0.740,   # B6  - Red Edge 2
    0.783,   # B7  - Red Edge 3
    0.842,   # B8  - NIR
    0.865,   # B8A - Narrow NIR
    0.945,   # B9  - Water vapour
    1.610,   # B11 - SWIR 1
    2.190,   # B12 - SWIR 2
]

# Wavelengths comuni per altri sensori
S1_WAVELENGTHS  = [5.405, 5.405]          # VV, VH  (SAR C-band)
RGB_WAVELENGTHS = [0.665, 0.560, 0.490]   # R, G, B

__all__ = [
    "DOFASeg",
    "DOFABackbone",
    "MLADecoder",
    "LinearDecoder",
    "DOFA_CONFIGS",
    "S2_WAVELENGTHS",
    "S1_WAVELENGTHS",
    "RGB_WAVELENGTHS",
]
