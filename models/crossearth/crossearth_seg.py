"""
crossearth – Full Segmentation Model
------------------------------------

Assembles:

  DINOv2WithRein  (dinov2_rein.py)
    ├── DINOv2 backbone  ← torch.hub (facebookresearch/dinov2)
    ├── optional RGBNIR patch embedding modification:
    │       Conv2d(3, embed_dim, 14, 14)
    │       → Conv2d(4, embed_dim, 14, 14)
    └── ReinAdapter      ← trainable PEFT adapter
        ↓
      [F1, F2, F3, F4] — (B, embed_dim, H/14, W/14) each

  MLADecoder / LinearDecoder  (decoder.py)  ← trainable
        ↓
  logits  (B, num_classes, H, W)

Recommended setup for RGBNIR remote sensing segmentation:

    - load pretrained DINOv2
    - replace patch embedding from 3 to 4 input channels
    - initialize NIR channel from mean RGB patch weights
    - freeze transformer backbone
    - train patch embedding + ReinAdapter + decoder

Dataset setup for your binary segmentation case:

    original labels:
        0 = invalid_pixel
        1 = sealed_soil
        2 = non_sealed_soil

    training labels:
        255 = ignore_index
        0   = sealed_soil
        1   = non_sealed_soil

    model:
        num_classes = 2

    loss:
        torch.nn.CrossEntropyLoss(ignore_index=255)

Usage
-----

RGB baseline:

    model = CrossEarthSeg.from_pretrained(
        variant="dinov2_vitl14_reg",
        num_classes=2,
        in_channels=3,
    )

    x = torch.randn(1, 3, 504, 504)
    logits = model(x)

RGBNIR:

    model = CrossEarthSeg.from_pretrained(
        variant="dinov2_vitl14_reg",
        num_classes=2,
        in_channels=4,
        patch_embed_init="rgb_mean",
        freeze_backbone=True,
        train_patch_embed=True,
    )

    x = torch.randn(1, 4, 504, 504)
    logits = model(x)

Notes
-----
H and W must be multiples of the DINOv2 patch size, usually 14.
Recommended sizes:
    504 = 36 × 14
    518 = 37 × 14
    1008 = 72 × 14
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .dinov2_rein import DINOv2WithRein, DINOV2_CONFIGS
from .decoder import MLADecoder, LinearDecoder


class CrossEarthSeg(nn.Module):
    """
    Full crossearth segmentation model.

    Parameters
    ----------
    variant:
        DINOv2 key, e.g. "dinov2_vitl14_reg".

    num_classes:
        Number of output segmentation classes.
        For your remapped task:
            0 = sealed_soil
            1 = non_sealed_soil
        so use num_classes=2.

    decoder:
        Decoder type:
            "mla"    -> MLADecoder, recommended
            "linear" -> lightweight linear decoder

    decoder_dim:
        Intermediate decoder channels.
        If None, selected automatically according to DINOv2 embed_dim.

    num_tokens:
        Learnable tokens per Rein layer.

    token_dim:
        Internal Rein token dimension.

    dropout:
        Dropout used in the decoder.

    in_channels:
        Number of input image channels:
            3 -> RGB
            4 -> RGBNIR

    patch_embed_init:
        Initialization for modified patch embedding when in_channels != 3.

        Options:
            "rgb_mean":
                Copy RGB weights and initialize additional channels as
                the mean of RGB weights. Recommended for RGBNIR with pretrained DINOv2.

            "zero":
                Copy RGB weights and initialize additional channels as zero.

            "random":
                Randomly initialize the whole new patch embedding.

    freeze_backbone:
        If True, freeze the DINOv2 transformer backbone.

    train_patch_embed:
        If True, keep patch embedding trainable even when the rest of
        the DINOv2 backbone is frozen.

        Recommended RGBNIR setup:
            freeze_backbone=True
            train_patch_embed=True
    """

    def __init__(
        self,
        variant: str = "dinov2_vitl14_reg",
        num_classes: int = 2,
        decoder: str = "mla",
        decoder_dim: Optional[int] = None,
        num_tokens: int = 100,
        token_dim: int = 256,
        dropout: float = 0.1,
        in_channels: int = 3,
        patch_embed_init: str = "rgb_mean",
        freeze_backbone: bool = True,
        train_patch_embed: bool = True,
    ):
        super().__init__()

        if variant not in DINOV2_CONFIGS:
            raise ValueError(
                f"Variante DINOv2 non supportata: {variant}. "
                f"Disponibili: {list(DINOV2_CONFIGS.keys())}"
            )

        if decoder not in {"mla", "linear"}:
            raise ValueError(
                f"Decoder non supportato: {decoder}. "
                "Usa 'mla' oppure 'linear'."
            )

        if in_channels <= 0:
            raise ValueError(
                f"in_channels deve essere > 0, ricevuto {in_channels}"
            )

        cfg = DINOV2_CONFIGS[variant]
        embed_dim = cfg["embed_dim"]
        patch_size = cfg["patch_size"]

        self.variant = variant
        self.num_classes = num_classes
        self.decoder_name = decoder
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.freeze_backbone_flag = freeze_backbone
        self.train_patch_embed = train_patch_embed

        # Default decoder dimension for each DINOv2 variant.
        _dec_dim = {
            384: 256,
            768: 256,
            1024: 512,
            1536: 768,
        }
        dec_dim = decoder_dim or _dec_dim.get(embed_dim, 512)

        # ------------------------------------------------------------------
        # Backbone + Rein
        # ------------------------------------------------------------------
        self.backbone_rein = DINOv2WithRein(
            variant=variant,
            num_tokens=num_tokens,
            token_dim=token_dim,
            in_channels=in_channels,
            patch_embed_init=patch_embed_init,
            freeze_backbone=freeze_backbone,
            train_patch_embed=train_patch_embed,
        )

        # ------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------
        if decoder == "mla":
            self.decoder = MLADecoder(
                embed_dim=embed_dim,
                decoder_dim=dec_dim,
                num_classes=num_classes,
                patch_size=patch_size,
                dropout=dropout,
            )
        else:
            self.decoder = LinearDecoder(
                embed_dim=embed_dim,
                num_scales=4,
                decoder_dim=dec_dim,
                num_classes=num_classes,
                patch_size=patch_size,
                dropout=dropout,
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Input tensor.

            RGB:
                shape = (B, 3, H, W)

            RGBNIR:
                shape = (B, 4, H, W)

            H and W must be multiples of patch_size, usually 14.

        Returns
        -------
        logits:
            Tensor with shape:

                (B, num_classes, H, W)
        """
        if x.ndim != 4:
            raise ValueError(
                f"x deve avere shape (B, C, H, W), ricevuto shape={x.shape}"
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Canali input incoerenti: modello creato con "
                f"in_channels={self.in_channels}, ma x ha C={x.shape[1]}."
            )

        features = self.backbone_rein(x)  # [F1, F2, F3, F4]
        logits = self.decoder(features)

        return logits

    # ------------------------------------------------------------------
    # Constructors / loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        variant: str = "dinov2_vitl14_reg",
        num_classes: int = 2,
        force_reload: bool = False,
        **kwargs,
    ) -> "CrossEarthSeg":
        """
        Creates the model and downloads pretrained DINOv2 from torch.hub.

        Example RGB:

            model = CrossEarthSeg.from_pretrained(
                variant="dinov2_vitl14_reg",
                num_classes=2,
                in_channels=3,
            )

        Example RGBNIR:

            model = CrossEarthSeg.from_pretrained(
                variant="dinov2_vitl14_reg",
                num_classes=2,
                in_channels=4,
                patch_embed_init="rgb_mean",
                freeze_backbone=True,
                train_patch_embed=True,
            )
        """
        model = cls(
            variant=variant,
            num_classes=num_classes,
            **kwargs,
        )

        model.backbone_rein.load_backbone(
            pretrained=True,
            force_reload=force_reload,
        )

        return model

    def load_backbone_checkpoint(self, ckpt_path: str) -> "CrossEarthSeg":
        """
        Loads the DINOv2 backbone from a local checkpoint.

        Example:

            model = CrossEarthSeg(
                variant="dinov2_vitl14_reg",
                num_classes=2,
                in_channels=4,
                patch_embed_init="rgb_mean",
            )

            model.load_backbone_checkpoint(
                "checkpoints/dinov2_vitl14_pretrain.pth"
            )
        """
        self.backbone_rein.load_backbone_from_checkpoint(ckpt_path)
        return self

    def load_rein_checkpoint(self, ckpt_path: str) -> "CrossEarthSeg":
        """
        Loads Rein + decoder weights from a checkpoint saved during training.

        This can also load patch embedding weights if they were saved in
        the checkpoint and the model architecture matches.

        Example:
            model.load_rein_checkpoint("checkpoints/crossearth_rgbnir.pth")
        """
        state = torch.load(ckpt_path, map_location="cpu")

        missing, unexpected = self.load_state_dict(state, strict=False)

        if missing:
            print(
                f"[crossearth] Missing weights ({len(missing)}): "
                f"{missing[:5]} ..."
            )

        if unexpected:
            print(
                f"[crossearth] Unexpected weights ({len(unexpected)}): "
                f"{unexpected[:5]} ..."
            )

        print(f"[crossearth] Checkpoint loaded from '{ckpt_path}'")

        return self

    # ------------------------------------------------------------------
    # Freezing helpers
    # ------------------------------------------------------------------

    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        Freezes or unfreezes the DINOv2 backbone.

        Note:
            If train_patch_embed=True, patch embedding remains trainable
            even when freeze=True.
        """
        self.freeze_backbone_flag = freeze
        self.backbone_rein.set_freeze_backbone(freeze)

    def unfreeze_last_blocks(self, n_blocks: int = 4) -> None:
        """
        Unfreezes only the last n DINOv2 transformer blocks.

        Useful after an initial stable training with frozen backbone.

        Example:
            model.unfreeze_last_blocks(n_blocks=4)
        """
        self.backbone_rein.unfreeze_last_blocks(n_blocks=n_blocks)

    # ------------------------------------------------------------------
    # Optimizer parameter groups
    # ------------------------------------------------------------------

    def parameter_groups(
        self,
        lr_patch_embed: float = 1e-5,
        lr_rein: float = 1e-4,
        lr_decoder: float = 1e-3,
        lr_backbone: float = 1e-5,
        weight_decay: float = 0.01,
    ) -> list:
        """
        Returns optimizer parameter groups.

        Recommended initial setup:

            optimizer = torch.optim.AdamW(
                model.parameter_groups(
                    lr_patch_embed=1e-5,
                    lr_rein=1e-4,
                    lr_decoder=1e-3,
                    weight_decay=0.01,
                )
            )

        Groups:
            - patch embedding, if trainable
            - Rein adapter
            - decoder
            - any additional trainable DINOv2 backbone parameters
              excluding patch embedding, e.g. after unfreeze_last_blocks()
        """
        groups = []

        backbone = self.backbone_rein.backbone

        # Patch embedding group.
        patch_params = []
        patch_param_ids = set()

        if backbone is not None and hasattr(backbone, "patch_embed"):
            for p in backbone.patch_embed.parameters():
                if p.requires_grad:
                    patch_params.append(p)
                    patch_param_ids.add(id(p))

        if patch_params:
            groups.append(
                {
                    "params": patch_params,
                    "lr": lr_patch_embed,
                    "weight_decay": weight_decay,
                    "name": "patch_embed",
                }
            )

        # Rein adapter group.
        rein_params = [
            p for p in self.backbone_rein.rein.parameters()
            if p.requires_grad
        ]

        if rein_params:
            groups.append(
                {
                    "params": rein_params,
                    "lr": lr_rein,
                    "weight_decay": weight_decay,
                    "name": "rein",
                }
            )

        # Decoder group.
        decoder_params = [
            p for p in self.decoder.parameters()
            if p.requires_grad
        ]

        if decoder_params:
            groups.append(
                {
                    "params": decoder_params,
                    "lr": lr_decoder,
                    "weight_decay": weight_decay,
                    "name": "decoder",
                }
            )

        # Additional trainable DINOv2 backbone parameters, excluding patch_embed.
        backbone_extra_params = []

        if backbone is not None:
            for p in backbone.parameters():
                if p.requires_grad and id(p) not in patch_param_ids:
                    backbone_extra_params.append(p)

        if backbone_extra_params:
            groups.append(
                {
                    "params": backbone_extra_params,
                    "lr": lr_backbone,
                    "weight_decay": weight_decay,
                    "name": "backbone_extra",
                }
            )

        return groups

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def count_parameters(self) -> dict:
        """
        Returns parameter counts for diagnostics.
        """
        def count_trainable(module: Optional[nn.Module]) -> int:
            if module is None:
                return 0
            return sum(
                p.numel() for p in module.parameters()
                if p.requires_grad
            )

        def count_total(module: Optional[nn.Module]) -> int:
            if module is None:
                return 0
            return sum(p.numel() for p in module.parameters())

        backbone = self.backbone_rein.backbone

        patch_embed_total = 0
        patch_embed_trainable = 0

        if backbone is not None and hasattr(backbone, "patch_embed"):
            patch_embed_total = count_total(backbone.patch_embed)
            patch_embed_trainable = count_trainable(backbone.patch_embed)

        return {
            "backbone_total": count_total(backbone),
            "backbone_trainable": count_trainable(backbone),
            "patch_embed_total": patch_embed_total,
            "patch_embed_trainable": patch_embed_trainable,
            "rein_total": count_total(self.backbone_rein.rein),
            "rein_trainable": count_trainable(self.backbone_rein.rein),
            "decoder_total": count_total(self.decoder),
            "decoder_trainable": count_trainable(self.decoder),
            "model_total": count_total(self),
            "model_trainable": count_trainable(self),
        }