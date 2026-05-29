"""
CrossEarthSeg – Full Segmentation Model
-----------------------------------------
Assembles a DINOv2 backbone with Rein PEFT adapter and a Mask2Former-style
decoder for dense semantic segmentation of multispectral remote sensing imagery.

Architecture
------------
    DINOv2WithRein  (dinov2_rein.py)
        ├── DINOv2 backbone          ← torch.hub (facebookresearch/dinov2)
        ├── Optional patch embedding replacement:
        │       Conv2d(3,  embed_dim, 14, 14)  →  Conv2d(N, embed_dim, 14, 14)
        └── ReinAdapter              ← lightweight trainable PEFT adapter
            ↓
          [F1, F2, F3, F4]  —  (B, embed_dim, H/14, W/14) each

    Mask2FormerDecoder  (decoder.py)  ← trainable, pure PyTorch (no mmseg)
        ↓
    logits  (B, num_classes, H, W)

Recommended setup for RGBNIR remote sensing segmentation
---------------------------------------------------------
    1. Load pretrained DINOv2 weights from torch.hub.
    2. Replace patch embedding from 3 to 4 input channels,
       initializing the NIR channel from the mean of RGB patch weights.
    3. Freeze the DINOv2 transformer backbone.
    4. Train patch embedding, ReinAdapter, and Mask2FormerDecoder.

Spatial constraints
-------------------
H and W must be multiples of the DINOv2 patch size (14).
Recommended input sizes:  504 (36×14),  518 (37×14),  1008 (72×14).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .dinov2_rein import DINOv2WithRein, DINOV2_CONFIGS
from .decoder import Mask2FormerDecoder


class CrossEarthSeg(nn.Module):
    """
    CrossEarth segmentation model: DINOv2 + Rein + Mask2Former decoder.

    Args:
        variant:           DINOv2 model key, e.g. 'dinov2_vitl14_reg'.
        num_classes:       Number of output segmentation classes.
        decoder_dim:       Intermediate decoder channels. Auto-selected from
                        embed_dim if None.
        num_tokens:        Number of learnable Rein tokens per layer.
        token_dim:         Internal Rein token dimension.
        dropout:           Dropout probability in the decoder.
        in_channels:       Number of input image channels (3=RGB, 4=RGBNIR).
        patch_embed_init:  Initialization strategy for the modified patch
                        embedding when in_channels != 3.
                        'rgb_mean': additional channels initialized as mean
                        of RGB weights (recommended for RGBNIR).
                        'zero': additional channels initialized to zero.
                        'random': full random re-initialization.
        freeze_backbone:   If True, freeze the DINOv2 transformer backbone.
        train_patch_embed: If True, keep patch embedding trainable even when
                        the backbone is frozen. Recommended for RGBNIR.
    """

    def __init__(
        self,
        variant: str = "dinov2_vitl14_reg",
        num_classes: int = 2,
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

        if in_channels <= 0:
            raise ValueError(
                f"in_channels deve essere > 0, ricevuto {in_channels}"
            )

        cfg = DINOV2_CONFIGS[variant]
        embed_dim = cfg["embed_dim"]
        patch_size = cfg["patch_size"]

        self.variant = variant
        self.num_classes = num_classes
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
        self.decoder = Mask2FormerDecoder(
            embed_dim=embed_dim,
            num_classes=num_classes,
            feat_channels=256,
            num_queries=100,
            num_decoder_layers=9,
            num_heads=8,
            dim_feedforward=2048,
            patch_size=patch_size,
            dropout=0.0,
        )
        
    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor of shape (B, in_channels, H, W).
            H and W must be multiples of patch_size (14).

        Returns:
            Logits of shape (B, num_classes, H, W).

        Raises:
            ValueError: If x does not have 4 dimensions or the channel count
                        does not match in_channels.
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
        Instantiate CrossEarthSeg and download pretrained DINOv2 from torch.hub.

        Args:
            variant:      DINOv2 model key, e.g. 'dinov2_vitl14_reg'.
            num_classes:  Number of output segmentation classes.
            force_reload: Force re-download of the torch.hub checkpoint.
            **kwargs:     Additional arguments forwarded to the constructor.

        Returns:
            CrossEarthSeg instance with pretrained DINOv2 backbone loaded.
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
        Load the DINOv2 backbone from a local checkpoint file.

        Args:
            ckpt_path: Path to the local checkpoint (.pth).

        Returns:
            self, for method chaining.
        """
        self.backbone_rein.load_backbone_from_checkpoint(ckpt_path)
        return self

    def load_rein_checkpoint(self, ckpt_path: str) -> "CrossEarthSeg":
        """
        Load Rein adapter and decoder weights from a training checkpoint.

        Patch embedding weights are also restored if present in the checkpoint
        and the architecture matches. Uses strict=False; missing and unexpected
        keys are reported to stdout.

        Args:
            ckpt_path: Path to the checkpoint saved during training.

        Returns:
            self, for method chaining.
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
        Freeze or unfreeze the DINOv2 backbone.

        Note: if train_patch_embed=True, the patch embedding remains trainable
        regardless of the freeze flag.

        Args:
            freeze: If True, disables gradient computation for the backbone.
                    If False, re-enables it (default True).
        """
        self.freeze_backbone_flag = freeze
        self.backbone_rein.set_freeze_backbone(freeze)

    def unfreeze_last_blocks(self, n_blocks: int = 4) -> None:
        """
        Unfreeze only the last n DINOv2 transformer blocks.

        Useful for gradual fine-tuning after an initial training phase with a
        fully frozen backbone.

        Args:
            n_blocks: Number of terminal transformer blocks to unfreeze (default 4).
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
        Return optimizer parameter groups with per-component learning rates.

        Groups (only included if they contain trainable parameters):
            patch_embed:    patch embedding (if trainable).
            rein:           Rein adapter tokens and MLP.
            decoder:        Mask2Former decoder.
            backbone_extra: any additional trainable backbone parameters,
                            e.g. after unfreeze_last_blocks(), excluding patch_embed.

        Args:
            lr_patch_embed: Learning rate for the patch embedding (default 1e-5).
            lr_rein:        Learning rate for the Rein adapter (default 1e-4).
            lr_decoder:     Learning rate for the decoder (default 1e-3).
            lr_backbone:    Learning rate for unfrozen backbone blocks (default 1e-5).
            weight_decay:   Weight decay applied to all groups (default 0.01).

        Returns:
            List of dicts suitable for passing directly to a PyTorch optimizer.
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
        Return parameter counts per component for diagnostics.

        Returns:
            Dict with keys: 'backbone_total', 'backbone_trainable',
            'patch_embed_total', 'patch_embed_trainable', 'rein_total',
            'rein_trainable', 'decoder_total', 'decoder_trainable',
            'model_total', 'model_trainable'.
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