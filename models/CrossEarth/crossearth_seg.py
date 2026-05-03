"""
CrossEarth – Full Segmentation Model
--------------------------------------
Assembles:
  DINOv2WithRein  (dinov2_rein.py)
    ├── DINOv2 backbone  ← torch.hub (facebookresearch/dinov2), fully FROZEN
    └── ReinAdapter      ← trainable PEFT adapter (~0.5–2M params)
        ↓  [F1, F2, F3, F4] — (B, embed_dim, H/14, W/14) each
  MLADecoder / LinearDecoder  (decoder.py)  ← trainable
        ↓
  logits  (B, num_classes, H, W)

What comes from libraries (zero custom code):
  - Full DINOv2 from torch.hub

What is custom:
  - ReinAdapter  (~60 lines) — PEFT adapter inspired by Rein (CVPR 2024)
  - DINOv2WithRein  (~80 lines) — hooks + multi-scale reshape
  - MLADecoder / LinearDecoder  (~70 lines) — lightweight decoder

Trainable parameters (ViT-L, default):
  Rein tokens:  num_layers × num_tokens × token_dim = 24 × 100 × 256 ≈ 600K
  Rein MLP:     ~1M
  MLA Decoder:  ~2M
  Total:        ~3.5M  vs  ~307M frozen backbone

Setup
-----
  Option A — automatic download (torch.hub):
      model = CrossEarthSeg.from_pretrained("dinov2_vitl14_reg", num_classes=14)

  Option B — local checkpoint (from https://dl.fbaipublicfiles.com/dinov2/):
      model = CrossEarthSeg(variant="dinov2_vitl14_reg", num_classes=14)
      model.load_backbone_checkpoint("checkpoints/dinov2_vitl14_pretrain.pth")

Usage
-----
    from models.CrossEarth import CrossEarthSeg

    model = CrossEarthSeg.from_pretrained("dinov2_vitl14_reg", num_classes=14)
    logits = model(x)   # x: (B, 3, H, W)  →  (B, num_classes, H, W)
    # H and W must be multiples of 14 (e.g. 518, 1022, 504, 1008...)
"""

import torch
import torch.nn as nn

from .dinov2_rein import DINOv2WithRein, DINOV2_CONFIGS
from .decoder     import MLADecoder, LinearDecoder


class CrossEarthSeg(nn.Module):
    """
    Parameters
    ----------
    variant      : str    DINOv2 key, e.g. "dinov2_vitl14_reg"
    num_classes  : int    segmentation classes
    decoder      : str    "mla" (recommended) | "linear"
    decoder_dim  : int    intermediate decoder channels (None = auto)
    num_tokens   : int    learnable tokens per Rein layer (default 100)
    token_dim    : int    internal Rein token dimension (default 256)
    dropout      : float  dropout in the decoder
    """

    def __init__(
        self,
        variant     : str   = "dinov2_vitl14_reg",
        num_classes : int   = 14,
        decoder     : str   = "mla",
        decoder_dim : int   = None,
        num_tokens  : int   = 100,
        token_dim   : int   = 256,
        dropout     : float = 0.1,
    ):
        super().__init__()
        cfg       = DINOV2_CONFIGS[variant]
        embed_dim = cfg["embed_dim"]
        patch_size = cfg["patch_size"]

        # Default decoder dimension for each variant
        _dec_dim = {384: 256, 768: 256, 1024: 512, 1536: 768}
        dec_dim = decoder_dim or _dec_dim.get(embed_dim, 512)

        # ── Backbone + Rein ───────────────────────────────────────────────
        self.backbone_rein = DINOv2WithRein(
            variant    = variant,
            num_tokens = num_tokens,
            token_dim  = token_dim,
        )

        # ── Decoder ───────────────────────────────────────────────────────
        if decoder == "mla":
            self.decoder = MLADecoder(
                embed_dim   = embed_dim,
                decoder_dim = dec_dim,
                num_classes = num_classes,
                patch_size  = patch_size,
                dropout     = dropout,
            )
        else:
            self.decoder = LinearDecoder(
                embed_dim   = embed_dim,
                num_scales  = 4,
                decoder_dim = dec_dim,
                num_classes = num_classes,
                patch_size  = patch_size,
                dropout     = dropout,
            )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W)
            H and W must be multiples of the patch_size (14).
            Suggested: 504 (36×14) or 518 (37×14) for 512-sized images.

        Returns
        -------
        logits : (B, num_classes, H, W)
        """
        features = self.backbone_rein(x)   # [F1, F2, F3, F4]
        logits   = self.decoder(features)
        return logits

    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        variant     : str = "dinov2_vitl14_reg",
        num_classes : int = 14,
        force_reload: bool = False,
        **kwargs,
    ) -> "CrossEarthSeg":
        """
        Creates the model and downloads pretrained DINOv2 from torch.hub.

        Example
        -------
            model = CrossEarthSeg.from_pretrained(
                "dinov2_vitl14_reg", num_classes=14
            )
        """
        model = cls(variant=variant, num_classes=num_classes, **kwargs)
        model.backbone_rein.load_backbone(
            pretrained=True, force_reload=force_reload
        )
        return model

    def load_backbone_checkpoint(self, ckpt_path: str) -> "CrossEarthSeg":
        """
        Loads the DINOv2 backbone from a local checkpoint.

        Example
        -------
            model = CrossEarthSeg(variant="dinov2_vitl14_reg", num_classes=14)
            model.load_backbone_checkpoint(
                "checkpoints/dinov2_vitl14_pretrain.pth"
            )
        """
        self.backbone_rein.load_backbone_from_checkpoint(ckpt_path)
        return self

    def load_rein_checkpoint(self, ckpt_path: str) -> "CrossEarthSeg":
        """
        Loads Rein + decoder weights from a checkpoint saved during training.

        Example
        -------
            model.load_rein_checkpoint("checkpoints/crossearth_rein_head.pth")
        """
        state = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            print(f"[CrossEarth] Missing weights ({len(missing)}): {missing[:3]} ...")
        print(f"[CrossEarth] Rein+decoder loaded from '{ckpt_path}'")
        return self

    # ------------------------------------------------------------------
    def freeze_backbone(self, freeze: bool = True):
        """The DINOv2 backbone is already frozen by default. This method
        allows freezing/unfreezing the Rein adapter as well if needed."""
        for p in self.backbone_rein.backbone.parameters():
            p.requires_grad = not freeze

    def parameter_groups(
        self,
        lr_rein     : float = 1e-4,
        lr_decoder  : float = 1e-3,
        weight_decay: float = 0.01,
    ) -> list:
        """
        Only the Rein adapter and decoder are trainable.
        The DINOv2 backbone is frozen → it does not need a parameter group.
        """
        return [
            {"params": self.backbone_rein.rein.parameters(),
             "lr": lr_rein, "weight_decay": weight_decay},
            {"params": self.decoder.parameters(),
             "lr": lr_decoder, "weight_decay": weight_decay},
        ]

    def count_parameters(self) -> dict:
        def n(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
        def total(m): return sum(p.numel() for p in m.parameters())
        backbone_params = total(self.backbone_rein.backbone) \
            if self.backbone_rein.backbone else 0
        return {
            "backbone (frozen)": backbone_params,
            "rein (trainable)" : n(self.backbone_rein.rein),
            "decoder (trainable)": n(self.decoder),
            "total trainable"  : n(self),
        }