"""
SegFormer + SAE + BRD  –  Full Segmentation Model
--------------------------------------------------
Assembla:
  SAEModule      (sae_module.py)        ← embedding spettrale
      ↓
  SegformerModel (HuggingFace)          ← backbone MiT pretrained
      ↓  [F1, F2, F3, F4]
  BRDDecoder     (brd_decoder.py)       ← boundary-refined decoder [opzionale]
      ↓  F5 (B, 64, H/4, W/4)
  ClassifierHead (segformer_decoder.py) ← logits full-resolution

  Se use_brd=False: [F1..F4] → SegFormerHead (All-MLP decoder originale)

Usage
-----
    from models.SegFormerSAE import SegFormerSAE

    # Con BRD (default)
    model = SegFormerSAE.from_pretrained("nvidia/mit-b2", in_channels=12, num_classes=14)

    # Senza BRD (decoder originale SegFormer)
    model = SegFormerSAE(variant="mit-b2", in_channels=12, num_classes=14, use_brd=False)
"""

import torch
import torch.nn as nn
from transformers import SegformerModel, SegformerConfig

from .sae_module        import SAEModule
from .brd_decoder       import BRDDecoder
from .segformer_decoder import SegFormerHead, ClassifierHead


# ---------------------------------------------------------------------------
# MiT variant configs
# ---------------------------------------------------------------------------

MiT_CONFIGS = {
    "mit-b0": dict(hidden_sizes=[32,  64,  160, 256], depths=[2, 2, 2, 2], num_attention_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1]),
    "mit-b1": dict(hidden_sizes=[64,  128, 320, 512], depths=[2, 2, 2, 2], num_attention_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1]),
    "mit-b2": dict(hidden_sizes=[64,  128, 320, 512], depths=[3, 4, 6, 3], num_attention_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1]),
    "mit-b3": dict(hidden_sizes=[64,  128, 320, 512], depths=[3, 4, 18,3], num_attention_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1]),
    "mit-b4": dict(hidden_sizes=[64,  128, 320, 512], depths=[3, 8, 27,3], num_attention_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1]),
    "mit-b5": dict(hidden_sizes=[64,  128, 320, 512], depths=[3, 6, 40,3], num_attention_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1]),
}

_DECODER_DIM = {
    "mit-b0": 128, "mit-b1": 256, "mit-b2": 256,
    "mit-b3": 256, "mit-b4": 512, "mit-b5": 512,
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SegFormerSAE(nn.Module):
    """
    Parameters
    ----------
    variant       : str    chiave MiT, es. 'mit-b2'
    in_channels   : int    bande spettrali in input (default 12)
    num_classes   : int    classi di segmentazione (default 14)
    use_brd       : bool   usa BRDDecoder (True) o SegFormerHead originale (False)
    decoder_dim   : int    dim fusione per SegFormerHead (None = auto, ignorato con BRD)
    sae_reduction : int    reduction ratio SAE channel attention (default 8)
    dropout       : float  dropout classifier head (default 0.1)
    drop_path     : float  stochastic depth encoder (default 0.1)
    """

    def __init__(
        self,
        variant      : str   = "mit-b2",
        in_channels  : int   = 12,
        num_classes  : int   = 14,
        use_brd      : bool  = True,
        decoder_dim  : int   = None,
        sae_reduction: int   = 8,
        dropout      : float = 0.1,
        drop_path    : float = 0.1,
    ):
        super().__init__()
        cfg_mit = MiT_CONFIGS[variant]
        c1      = cfg_mit["hidden_sizes"][0]
        dec_dim = decoder_dim or _DECODER_DIM[variant]

        # ── 1. SAE ────────────────────────────────────────────────────────
        self.sae = SAEModule(
            in_channels = in_channels,
            embed_dim   = c1,
            reduction   = sae_reduction,
        )

        # ── 2. MiT Encoder (HuggingFace) ─────────────────────────────────
        hf_config = SegformerConfig(
            num_channels        = c1,
            hidden_sizes        = cfg_mit["hidden_sizes"],
            depths              = cfg_mit["depths"],
            num_attention_heads = cfg_mit["num_attention_heads"],
            sr_ratios           = cfg_mit["sr_ratios"],
            drop_path_rate      = drop_path,
        )
        self.encoder = SegformerModel(hf_config)

        # ── 3. Decoder ────────────────────────────────────────────────────
        self.use_brd = use_brd
        if use_brd:
            brd_out = cfg_mit["hidden_sizes"][0]   # F5 channels = C1 = 64
            self.brd     = BRDDecoder(
                encoder_channels = cfg_mit["hidden_sizes"],
                out_channels     = brd_out,
            )
            self.cls_head = ClassifierHead(
                in_channels = brd_out,
                num_classes = num_classes,
                dropout     = dropout,
            )
        else:
            self.seg_head = SegFormerHead(
                in_channels = cfg_mit["hidden_sizes"],
                embed_dim   = dec_dim,
                num_classes = num_classes,
                dropout     = dropout,
            )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SAE
        x = self.sae(x)                                    # (B, C1, H, W)

        # Encoder
        enc_out  = self.encoder(pixel_values=x, output_hidden_states=True)
        features = list(enc_out.hidden_states)             # [F1, F2, F3, F4]

        # Decoder
        if self.use_brd:
            f5     = self.brd(features)                    # (B, C1, H/4, W/4)
            logits = self.cls_head(f5)                     # (B, num_classes, H, W)
        else:
            logits = self.seg_head(features)               # (B, num_classes, H, W)

        return logits

    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        hf_model_name : str,
        in_channels   : int   = 12,
        num_classes   : int   = 14,
        **kwargs,
    ) -> "SegFormerSAE":
        """
        Crea il modello e carica i pesi pretrained dell'encoder da HuggingFace.
        strict=False perché num_channels cambia da 3 → C1 (output SAE).

        Esempio
        -------
            model = SegFormerSAE.from_pretrained(
                "nvidia/mit-b2", in_channels=12, num_classes=14
            )
        """
        variant = hf_model_name.split("/")[-1]
        model   = cls(variant=variant, in_channels=in_channels,
                      num_classes=num_classes, **kwargs)

        pretrained = SegformerModel.from_pretrained(hf_model_name)
        missing, _ = model.encoder.load_state_dict(
            pretrained.state_dict(), strict=False
        )
        if missing:
            print(f"[SegFormerSAE] Pesi mancanti ({len(missing)}): {missing[:3]} ...")
        print(f"[SegFormerSAE] Encoder caricato da '{hf_model_name}'")
        return model

    # ------------------------------------------------------------------
    def freeze_encoder(self, freeze: bool = True):
        for p in self.encoder.parameters():
            p.requires_grad = not freeze

    def parameter_groups(
        self,
        lr_encoder  : float = 6e-5,
        lr_decoder  : float = 6e-4,
        lr_sae      : float = 6e-4,
        weight_decay: float = 0.01,
    ) -> list:
        decoder_params = (
            list(self.brd.parameters()) + list(self.cls_head.parameters())
            if self.use_brd else list(self.seg_head.parameters())
        )
        return [
            {"params": self.sae.parameters(),     "lr": lr_sae,    "weight_decay": weight_decay},
            {"params": self.encoder.parameters(), "lr": lr_encoder, "weight_decay": weight_decay},
            {"params": decoder_params,            "lr": lr_decoder, "weight_decay": weight_decay},
        ]

    def count_parameters(self) -> dict:
        def n(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
        dec = (n(self.brd) + n(self.cls_head)) if self.use_brd else n(self.seg_head)
        return {
            "sae"    : n(self.sae),
            "encoder": n(self.encoder),
            "decoder": dec,
            "total"  : n(self),
        }
