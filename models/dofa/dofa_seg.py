"""
DOFASeg – Full Segmentation Model
----------------------------------
Assembles:
  DOFABackbone  (dofa_backbone.py)
    └── dofa from torchgeo  ← pretrained wave-dynamic backbone, optionally frozen
        ↓  [F1, F2, F3, F4] — (B, embed_dim, H/p, W/p) each
  MLADecoder / LinearDecoder  (decoder.py)
        ↓
  logits  (B, num_classes, H, W)

What comes from torchgeo (zero custom code):
  - Full dofa (dofa_base_patch16_224, dofa_large_patch16_224)
  - DOFABase16_Weights.DOFA_MAE / DOFALarge16_Weights.DOFA_MAE
  - DOFAEmbedding (wave-dynamic patch embedding, inside torchgeo)

What is custom (~80 lines):
  - Forward hooks to extract intermediate features  (dofa_backbone.py)
  - MLADecoder / LinearDecoder  (decoder.py)

Trainable parameters (base, frozen backbone):
  MLA Decoder: ~3–5M  vs  ~86M frozen backbone

Trainable parameters (base, unfrozen backbone):
  Everything: ~89M

Usage
-----
    from models.dofa import DOFASeg

    # With pretrained frozen backbone (fine-tune decoder only)
    model = DOFASeg(variant="base", num_classes=14, pretrained=True, freeze_backbone=True)
    logits = model(x, wavelengths)  # x: (B, C, H, W)

    # With unfrozen backbone (fine-tune everything)
    model = DOFASeg(variant="large", num_classes=14, pretrained=True, freeze_backbone=False)

Input notes
-----------
  - H and W must be multiples of patch_size (16 for base/large, 14 for huge).
  - MAE weights were trained with img_size=224 → use 224×224 for maximum
    compatibility, or any multiple of 16, e.g. 256, 512.
  - wavelengths: list[float]  — wavelengths in µm, one per band.
    Examples:
      S2 (9 bands):  [0.665, 0.56, 0.49, 0.705, 0.74, 0.783, 0.842, 1.61, 2.19]
      S2 (12 bands): S2_WAVELENGTHS  (from __init__.py)
      S1 (2 bands):  [5.405, 5.405]
      RGB:           [0.665, 0.56, 0.49]
"""

import torch
import torch.nn as nn

from .dofa_backbone import DOFABackbone, DOFA_CONFIGS
from .decoder       import MLADecoder, LinearDecoder


# Recommended decoder dimension for each variant
_DECODER_DIM = {
    "small": 128,
    "base" : 256,
    "large": 512,
    "huge" : 768,
}


class DOFASeg(nn.Module):
    """
    Parameters
    ----------
    variant         : "small" | "base" | "large" | "huge"
    num_classes     : int    segmentation classes
    pretrained      : bool   pretrained MAE weights from torchgeo (base and large)
    freeze_backbone : bool   freeze the backbone after loading
    decoder         : str    "mla" (default) | "linear"
    decoder_dim     : int    intermediate decoder channels (None = auto)
    out_indices     : list   block indices for multi-scale features (None = default)
    dropout         : float  dropout in the decoder
    """

    def __init__(
        self,
        variant        : str   = "base",
        num_classes    : int   = 14,
        pretrained     : bool  = True,
        freeze_backbone: bool  = True,
        decoder        : str   = "mla",
        decoder_dim    : int   = None,
        out_indices    : list  = None,
        dropout        : float = 0.1,
    ):
        super().__init__()
        cfg       = DOFA_CONFIGS[variant]
        embed_dim = cfg["embed_dim"]
        patch_size = cfg["patch_size"]
        dec_dim   = decoder_dim or _DECODER_DIM[variant]

        # ── Backbone ──────────────────────────────────────────────────────
        self.backbone = DOFABackbone(
            variant     = variant,
            pretrained  = pretrained,
            out_indices = out_indices,
        )
        if freeze_backbone:
            self.backbone.freeze(True)

        # ── Decoder ───────────────────────────────────────────────────────
        dec_cls = MLADecoder if decoder == "mla" else LinearDecoder
        self.decoder = dec_cls(
            embed_dim   = embed_dim,
            decoder_dim = dec_dim,
            num_classes = num_classes,
            patch_size  = patch_size,
            dropout     = dropout,
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, wavelengths: list) -> torch.Tensor:
        """
        Parameters
        ----------
        x           : (B, C, H, W)   multispectral image
        wavelengths : list[float]    wavelengths in µm, one per band
                                     len(wavelengths) must be == C

        Returns
        -------
        logits : (B, num_classes, H, W)
        """
        features = self.backbone(x, wavelengths)   # [F1, F2, F3, F4]
        return self.decoder(features)

    # ------------------------------------------------------------------
    def freeze_backbone(self, freeze: bool = True):
        self.backbone.freeze(freeze)

    def parameter_groups(
        self,
        lr_backbone : float = 1e-5,
        lr_decoder  : float = 1e-4,
        weight_decay: float = 0.05,
    ) -> list:
        """
        Separate learning rates: backbone for slow fine-tuning, if unfrozen,
        and decoder from scratch, always trainable.
        """
        return [
            {"params": self.backbone.parameters(), "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": self.decoder.parameters(),  "lr": lr_decoder,  "weight_decay": weight_decay},
        ]

    def count_parameters(self) -> dict:
        def n_trainable(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
        def n_total(m):     return sum(p.numel() for p in m.parameters())
        return {
            "backbone (total)"    : n_total(self.backbone),
            "backbone (trainable)": n_trainable(self.backbone),
            "decoder (trainable)" : n_trainable(self.decoder),
            "total trainable"     : n_trainable(self),
        }