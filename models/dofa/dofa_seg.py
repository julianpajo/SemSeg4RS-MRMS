"""
DOFASeg – Full Segmentation Model
----------------------------------
Assembles a wave-dynamic DOFA backbone with a lightweight segmentation
decoder for multi-spectral remote sensing imagery.

Components
----------
DOFABackbone  (dofa_backbone.py)
    Wraps the torchgeo DOFA vision transformer and extracts intermediate
    features [F1, F2, F3, F4] via forward hooks.

MLADecoder / LinearDecoder  (decoder.py)
    Fuses multi-scale features and produces full-resolution logits.

What comes from torchgeo (no custom code):
    - DOFA model variants (dofa_base_patch16_224, dofa_large_patch16_224)
    - Pretrained MAE weights (DOFABase16_Weights, DOFALarge16_Weights)
    - DOFAEmbedding (wave-dynamic patch embedding)

What is custom (~80 lines):
    - Forward hooks for intermediate feature extraction  (dofa_backbone.py)
    - MLADecoder / LinearDecoder  (decoder.py)

Approximate parameter counts
-----------------------------
    Base backbone (frozen):   ~86M frozen  +  ~3–5M trainable decoder
    Base backbone (unfrozen): ~89M fully trainable

Input requirements
------------------
    - Spatial dimensions H and W must be multiples of patch_size
      (16 for base/large, 14 for huge).
    - MAE weights were trained at 224×224; use 224 or any multiple of 16
      (e.g. 256, 512) for best compatibility.
    - wavelengths: list[float] — one wavelength in µm per input band.

      Examples:
        S2  9-band:  [0.665, 0.56, 0.49, 0.705, 0.74, 0.783, 0.842, 1.61, 2.19]
        S2 12-band:  S2_WAVELENGTHS  (from __init__.py)
        S1  2-band:  [5.405, 5.405]
        RGB:         [0.665, 0.56, 0.49]

Usage
-----
    from models.dofa import DOFASeg

    # Fine-tune decoder only (frozen backbone)
    model = DOFASeg(variant="base", num_classes=14, pretrained=True, freeze_backbone=True)
    logits = model(x, wavelengths)   # x: (B, C, H, W)

    # Fine-tune everything (unfrozen backbone)
    model = DOFASeg(variant="large", num_classes=14, pretrained=True, freeze_backbone=False)
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
    Wave-dynamic segmentation model built on DOFA + a lightweight decoder.

    Args:
        variant:         DOFA backbone size, one of
                        ['small', 'base', 'large', 'huge'] (default 'base').
        num_classes:     Number of output segmentation classes (default 14).
        pretrained:      Load MAE pretrained weights from torchgeo (default True).
                        Only available for 'base' and 'large'.
        freeze_backbone: Freeze backbone parameters after loading (default True).
        decoder:         Decoder architecture, 'mla' (default) or 'linear'.
        decoder_dim:     Intermediate channel dimension in the decoder.
                        Falls back to the variant default from _DECODER_DIM if None.
        out_indices:     Backbone block indices from which to extract features.
                        None uses the variant default.
        dropout:         Dropout probability in the decoder (default 0.1).
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
        """
        Freeze or unfreeze all backbone parameters.

        Args:
            freeze: If True, disables gradient computation for the backbone.
                    If False, re-enables it (default True).
        """
        self.backbone.freeze(freeze)

    def parameter_groups(
        self,
        lr_backbone : float = 1e-5,
        lr_decoder  : float = 1e-4,
        weight_decay: float = 0.05,
    ) -> list:
        """
        Return parameter groups with separate learning rates for the backbone
        and decoder, suitable for passing directly to a PyTorch optimizer.

        The backbone uses a lower learning rate for slow fine-tuning when unfrozen;
        the decoder is always trainable and trained from scratch at a higher rate.

        Args:
            lr_backbone:  Learning rate for the backbone (default 1e-5).
            lr_decoder:   Learning rate for the decoder (default 1e-4).
            weight_decay: Weight decay applied to all groups (default 0.05).

        Returns:
            List of two dicts: one for the backbone and one for the decoder.
        """
        return [
            {"params": self.backbone.parameters(), "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": self.decoder.parameters(),  "lr": lr_decoder,  "weight_decay": weight_decay},
        ]

    def count_parameters(self) -> dict:
        """
        Count total and trainable parameters per component.

        Returns:
            Dict with keys:
                'backbone (total)'     – all backbone parameters,
                'backbone (trainable)' – unfrozen backbone parameters,
                'decoder (trainable)'  – decoder parameters,
                'total trainable'      – all trainable parameters in the model.
        """
        def n_trainable(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
        def n_total(m):     return sum(p.numel() for p in m.parameters())
        return {
            "backbone (total)"    : n_total(self.backbone),
            "backbone (trainable)": n_trainable(self.backbone),
            "decoder (trainable)" : n_trainable(self.decoder),
            "total trainable"     : n_trainable(self),
        }