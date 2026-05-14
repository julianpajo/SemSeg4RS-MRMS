"""
SegFormer + SAE + BRD  –  Full Segmentation Model
--------------------------------------------------
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
    "mit-b0": dict(hidden_sizes=[32, 64, 160, 256], depths=[2, 2, 2, 2], num_attention_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1]),
    "mit-b1": dict(hidden_sizes=[64, 128, 320, 512], depths=[2, 2, 2, 2], num_attention_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1]),
    "mit-b2": dict(hidden_sizes=[64, 128, 320, 512], depths=[3, 4, 6, 3], num_attention_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1]),
    "mit-b3": dict(hidden_sizes=[64, 128, 320, 512], depths=[3, 4, 18, 3], num_attention_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1]),
    "mit-b4": dict(hidden_sizes=[64, 128, 320, 512], depths=[3, 8, 27, 3], num_attention_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1]),
    "mit-b5": dict(hidden_sizes=[64, 128, 320, 512], depths=[3, 6, 40, 3], num_attention_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1]),
}

_DECODER_DIM = {
    "mit-b0": 128,
    "mit-b1": 256,
    "mit-b2": 256,
    "mit-b3": 256,
    "mit-b4": 512,
    "mit-b5": 512,
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SegFormerSAE(nn.Module):
    def __init__(
        self,
        variant: str = "mit-b2",
        in_channels: int = 12,
        num_classes: int = 14,
        use_brd: bool = True,
        decoder_dim: int = None,
        sae_reduction: int = 8,
        dropout: float = 0.1,
        drop_path: float = 0.1,
    ):
        super().__init__()

        if variant not in MiT_CONFIGS:
            raise ValueError(
                f"Unknown MiT variant '{variant}'. "
                f"Available: {list(MiT_CONFIGS.keys())}"
            )

        cfg_mit = MiT_CONFIGS[variant]
        c1 = cfg_mit["hidden_sizes"][0]
        dec_dim = decoder_dim or _DECODER_DIM[variant]

        self.variant = variant
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_brd = use_brd

        # 1. SAE: input multispectral/RGBNIR -> C1 feature map
        self.sae = SAEModule(
            in_channels=in_channels,
            embed_dim=c1,
            reduction=sae_reduction,
        )

        # 2. MiT encoder: receives SAE output, not raw RGB
        hf_config = SegformerConfig(
            num_channels=c1,
            hidden_sizes=cfg_mit["hidden_sizes"],
            depths=cfg_mit["depths"],
            num_attention_heads=cfg_mit["num_attention_heads"],
            sr_ratios=cfg_mit["sr_ratios"],
            drop_path_rate=drop_path,
        )
        self.encoder = SegformerModel(hf_config)

        # 3. Decoder
        if use_brd:
            brd_out = cfg_mit["hidden_sizes"][0]
            self.brd = BRDDecoder(
                encoder_channels=cfg_mit["hidden_sizes"],
                out_channels=brd_out,
            )
            self.cls_head = ClassifierHead(
                in_channels=brd_out,
                num_classes=num_classes,
                dropout=dropout,
            )
        else:
            self.seg_head = SegFormerHead(
                in_channels=cfg_mit["hidden_sizes"],
                embed_dim=dec_dim,
                num_classes=num_classes,
                dropout=dropout,
            )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:      (B, in_channels, H, W)
        return: (B, num_classes, H, W)
        """
        input_hw = x.shape[-2:]

        # SAE
        x = self.sae(x)  # (B, C1, H, W)

        # Encoder
        enc_out = self.encoder(pixel_values=x, output_hidden_states=True)
        features = list(enc_out.hidden_states)  # [F1, F2, F3, F4]

        # Decoder
        if self.use_brd:
            f5 = self.brd(features)
            logits = self.cls_head(f5)
        else:
            logits = self.seg_head(features)

        # Safety: ensure full-resolution logits
        if logits.shape[-2:] != input_hw:
            logits = torch.nn.functional.interpolate(
                logits,
                size=input_hw,
                mode="bilinear",
                align_corners=False,
            )

        return logits

    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        hf_model_name: str,
        in_channels: int = 12,
        num_classes: int = 14,
        **kwargs,
    ) -> "segformer_sae":
        """
        Creates segformer_sae and loads compatible pretrained MiT encoder weights.

        Important:
        The HuggingFace MiT checkpoint expects RGB input in the first patch
        embedding:

            [C1, 3, 7, 7]

        This model receives SAE features instead:

            [C1, C1, 7, 7]

        Therefore, the first patch embedding weights are intentionally skipped.
        """
        variant = hf_model_name.split("/")[-1]

        model = cls(
            variant=variant,
            in_channels=in_channels,
            num_classes=num_classes,
            **kwargs,
        )

        pretrained = SegformerModel.from_pretrained(hf_model_name)
        state_dict = pretrained.state_dict()

        # Drop incompatible first patch embedding.
        # HF pretrained: RGB -> C1
        # This model: SAE(C input -> C1), then C1 -> C1
        keys_to_drop = [
            k for k in state_dict.keys()
            if k.startswith("encoder.patch_embeddings.0.proj.")
        ]

        for k in keys_to_drop:
            state_dict.pop(k)

        missing, unexpected = model.encoder.load_state_dict(
            state_dict,
            strict=False,
        )

        if keys_to_drop:
            print(
                "[segformer_sae] Skipped incompatible first patch embedding: "
                f"{keys_to_drop}"
            )

        if missing:
            print(
                f"[segformer_sae] Missing weights ({len(missing)}): "
                f"{missing[:8]} ..."
            )

        if unexpected:
            print(
                f"[segformer_sae] Unexpected weights ({len(unexpected)}): "
                f"{unexpected[:8]} ..."
            )

        print(f"[segformer_sae] Encoder loaded from '{hf_model_name}'")
        return model

    # ------------------------------------------------------------------
    def freeze_encoder(self, freeze: bool = True):
        for p in self.encoder.parameters():
            p.requires_grad = not freeze

    # ------------------------------------------------------------------
    def parameter_groups(
        self,
        lr_encoder: float = 6e-5,
        lr_decoder: float = 6e-4,
        lr_sae: float = 6e-4,
        weight_decay: float = 0.01,
    ) -> list:
        decoder_params = (
            list(self.brd.parameters()) + list(self.cls_head.parameters())
            if self.use_brd
            else list(self.seg_head.parameters())
        )

        return [
            {
                "params": self.sae.parameters(),
                "lr": lr_sae,
                "weight_decay": weight_decay,
            },
            {
                "params": self.encoder.parameters(),
                "lr": lr_encoder,
                "weight_decay": weight_decay,
            },
            {
                "params": decoder_params,
                "lr": lr_decoder,
                "weight_decay": weight_decay,
            },
        ]

    # ------------------------------------------------------------------
    def count_parameters(self) -> dict:
        def n(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        decoder_count = (
            n(self.brd) + n(self.cls_head)
            if self.use_brd
            else n(self.seg_head)
        )

        return {
            "sae": n(self.sae),
            "encoder": n(self.encoder),
            "decoder": decoder_count,
            "total": n(self),
        }