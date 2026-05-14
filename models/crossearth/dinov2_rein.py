"""
DINOv2 Backbone + Rein
----------------------

Wraps the DINOv2 backbone, loaded from torch.hub, and injects
the ReinAdapter after selected transformer blocks through forward hooks.

The wrapper supports both:

    RGB input:
        x.shape = (B, 3, H, W)

    RGBNIR input:
        x.shape = (B, 4, H, W)

For RGBNIR, the DINOv2 patch embedding is modified from:

    Conv2d(3, embed_dim, kernel_size=14, stride=14)

to:

    Conv2d(4, embed_dim, kernel_size=14, stride=14)

The fourth channel can be initialized from the mean of the original
RGB patch-embedding weights.

Recommended setup for RGBNIR remote sensing segmentation:

    - load pretrained DINOv2
    - replace patch embedding 3 -> 4 channels
    - freeze transformer backbone
    - train patch embedding + ReinAdapter + decoder

DINOv2 variants available on torch.hub:

    "dinov2_vits14"      -> ViT-S/14  embed=384,  layers=12
    "dinov2_vitb14"      -> ViT-B/14  embed=768,  layers=12
    "dinov2_vitl14"      -> ViT-L/14  embed=1024, layers=24
    "dinov2_vitg14"      -> ViT-G/14  embed=1536, layers=40

Versions with registers:

    "dinov2_vits14_reg"
    "dinov2_vitb14_reg"
    "dinov2_vitl14_reg"
    "dinov2_vitg14_reg"
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import List, Optional

import torch
import torch.nn as nn

from .rein_adapter import ReinAdapter


# ---------------------------------------------------------------------------
# DINOv2 configs
# ---------------------------------------------------------------------------

DINOV2_CONFIGS = {
    "dinov2_vits14": dict(embed_dim=384, num_layers=12, patch_size=14),
    "dinov2_vitb14": dict(embed_dim=768, num_layers=12, patch_size=14),
    "dinov2_vitl14": dict(embed_dim=1024, num_layers=24, patch_size=14),
    "dinov2_vitg14": dict(embed_dim=1536, num_layers=40, patch_size=14),

    "dinov2_vits14_reg": dict(embed_dim=384, num_layers=12, patch_size=14),
    "dinov2_vitb14_reg": dict(embed_dim=768, num_layers=12, patch_size=14),
    "dinov2_vitl14_reg": dict(embed_dim=1024, num_layers=24, patch_size=14),
    "dinov2_vitg14_reg": dict(embed_dim=1536, num_layers=40, patch_size=14),
}


# Layer indices from which multi-scale features are extracted.
# These are uniformly spaced checkpoints over the transformer depth.
_MULTISCALE_LAYERS = {
    12: [2, 5, 8, 11],
    24: [5, 11, 17, 23],
    40: [9, 19, 29, 39],
}


# ---------------------------------------------------------------------------
# Patch embedding replacement
# ---------------------------------------------------------------------------

def replace_patch_embed_input_channels(
    backbone: nn.Module,
    in_channels: int = 4,
    init_mode: str = "rgb_mean",
) -> nn.Module:
    """
    Replaces DINOv2 patch embedding to support a different number of
    input channels.

    Original DINOv2 patch embedding:

        Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=14,
            stride=14,
        )

    RGBNIR patch embedding:

        Conv2d(
            in_channels=4,
            out_channels=embed_dim,
            kernel_size=14,
            stride=14,
        )

    Parameters
    ----------
    backbone:
        DINOv2 backbone returned by torch.hub.

    in_channels:
        Desired number of input channels.
        Use 3 for RGB, 4 for RGBNIR.

    init_mode:
        Initialization strategy for the new patch embedding.

        "rgb_mean":
            Copies the original RGB weights and initializes additional
            channels as the mean of the RGB weights.

        "zero":
            Copies the original RGB weights and initializes additional
            channels as zero.

        "random":
            Randomly initializes the whole new patch embedding.
            This is mainly useful for true training from scratch.

    Returns
    -------
    backbone:
        Backbone with modified patch embedding.
    """
    if in_channels <= 0:
        raise ValueError(f"in_channels deve essere > 0, ricevuto {in_channels}")

    if not hasattr(backbone, "patch_embed"):
        raise AttributeError("Il backbone non ha attributo 'patch_embed'.")

    if not hasattr(backbone.patch_embed, "proj"):
        raise AttributeError("backbone.patch_embed non ha attributo 'proj'.")

    old_proj = backbone.patch_embed.proj

    if not isinstance(old_proj, nn.Conv2d):
        raise TypeError(
            "backbone.patch_embed.proj dovrebbe essere nn.Conv2d, "
            f"ricevuto {type(old_proj)}"
        )

    old_in_channels = old_proj.in_channels

    if old_in_channels == in_channels:
        return backbone

    new_proj = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
        dilation=old_proj.dilation,
        groups=old_proj.groups,
        bias=old_proj.bias is not None,
        padding_mode=old_proj.padding_mode,
    )

    with torch.no_grad():
        if init_mode == "random":
            nn.init.trunc_normal_(new_proj.weight, std=0.02)

            if new_proj.bias is not None:
                nn.init.zeros_(new_proj.bias)

        elif init_mode in {"rgb_mean", "zero"}:
            new_proj.weight.zero_()

            channels_to_copy = min(old_in_channels, in_channels)

            new_proj.weight[:, :channels_to_copy, :, :] = (
                old_proj.weight[:, :channels_to_copy, :, :]
            )

            if in_channels > old_in_channels:
                if init_mode == "rgb_mean":
                    mean_weight = old_proj.weight.mean(dim=1, keepdim=True)

                    for c in range(old_in_channels, in_channels):
                        new_proj.weight[:, c:c + 1, :, :] = mean_weight

                elif init_mode == "zero":
                    pass

            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)

        else:
            raise ValueError(
                f"init_mode non valido: {init_mode}. "
                "Usa: 'rgb_mean', 'zero', oppure 'random'."
            )

    backbone.patch_embed.proj = new_proj

    return backbone


# ---------------------------------------------------------------------------
# DINOv2 + Rein wrapper
# ---------------------------------------------------------------------------

class DINOv2WithRein(nn.Module):
    """
    Frozen or trainable DINOv2 backbone + Rein PEFT adapter.

    Parameters
    ----------
    variant:
        torch.hub model name, e.g. "dinov2_vitl14_reg".

    num_tokens:
        Number of learnable Rein tokens per layer.

    token_dim:
        Internal Rein token dimension.

    out_indices:
        Transformer block indices from which features are extracted.
        If None, default indices are used according to the DINOv2 depth.

    in_channels:
        Number of input image channels.

        Use:
            3 -> RGB
            4 -> RGBNIR

    patch_embed_init:
        Initialization strategy used when in_channels != 3.

        "rgb_mean":
            Copies RGB patch weights and initializes the NIR channel
            as the mean of RGB weights.

        "zero":
            Copies RGB patch weights and initializes the NIR channel to zero.

        "random":
            Random initialization of the new patch embedding.

    freeze_backbone:
        If True, freezes the DINOv2 transformer backbone.

    train_patch_embed:
        If True, keeps the patch embedding trainable even when the rest
        of the backbone is frozen.

        Recommended for RGBNIR:
            freeze_backbone=True
            train_patch_embed=True
    """

    def __init__(
        self,
        variant: str = "dinov2_vitl14_reg",
        num_tokens: int = 100,
        token_dim: int = 256,
        out_indices: Optional[List[int]] = None,
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
            raise ValueError(f"in_channels deve essere > 0, ricevuto {in_channels}")

        cfg = DINOV2_CONFIGS[variant]

        self.embed_dim = cfg["embed_dim"]
        self.num_layers = cfg["num_layers"]
        self.patch_size = cfg["patch_size"]
        self._variant = variant

        self.in_channels = in_channels
        self.patch_embed_init = patch_embed_init
        self.freeze_backbone_flag = freeze_backbone
        self.train_patch_embed = train_patch_embed

        # Backbone placeholder. It is populated by load_backbone()
        # or load_backbone_from_checkpoint().
        self.backbone: Optional[nn.Module] = None

        # Rein adapter, trainable.
        self.rein = ReinAdapter(
            num_layers=self.num_layers,
            embed_dim=self.embed_dim,
            num_tokens=num_tokens,
            token_dim=token_dim,
        )

        # Layers from which multi-scale features are extracted.
        self.out_indices = out_indices or _MULTISCALE_LAYERS[self.num_layers]

        # Storage for intermediate features captured by hooks.
        self._features: list = []
        self._hooks: list = []

    # ------------------------------------------------------------------
    # Backbone loading
    # ------------------------------------------------------------------

    def load_backbone(
        self,
        pretrained: bool = True,
        force_reload: bool = False,
    ) -> "DINOv2WithRein":
        """
        Loads DINOv2 from torch.hub.

        If in_channels != 3, the patch embedding is replaced after loading.

        For RGBNIR with pretrained DINOv2:

            pretrained=True
            in_channels=4
            patch_embed_init="rgb_mean"
            freeze_backbone=True
            train_patch_embed=True
        """
        print(f"[crossearth] Loading backbone '{self._variant}' from torch.hub...")

        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            self._variant,
            pretrained=pretrained,
            force_reload=force_reload,
        )

        self._maybe_replace_patch_embed()
        self._set_backbone_trainability()
        self._register_hooks()

        print("[crossearth] Backbone loaded.")
        print(
            f"[crossearth] freeze_backbone={self.freeze_backbone_flag}, "
            f"train_patch_embed={self.train_patch_embed}, "
            f"in_channels={self.in_channels}"
        )

        return self

    def load_backbone_from_checkpoint(
        self,
        ckpt_path: str,
    ) -> "DINOv2WithRein":
        """
        Loads DINOv2 from a local checkpoint.

        The architecture is first created through torch.hub with
        pretrained=False, then the checkpoint is loaded.

        If in_channels != 3, the patch embedding is replaced after
        loading the checkpoint.
        """
        print(f"[crossearth] Loading backbone from '{ckpt_path}'...")

        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            self._variant,
            pretrained=False,
        )

        state = torch.load(ckpt_path, map_location="cpu")

        if isinstance(state, dict) and "model" in state:
            state = state["model"]

        missing, unexpected = self.backbone.load_state_dict(state, strict=False)

        if missing:
            print(f"[crossearth] Missing weights ({len(missing)}): {missing[:3]} ...")

        if unexpected:
            print(
                f"[crossearth] Unexpected weights ({len(unexpected)}): "
                f"{unexpected[:3]} ..."
            )

        self._maybe_replace_patch_embed()
        self._set_backbone_trainability()
        self._register_hooks()

        print("[crossearth] Backbone loaded from checkpoint.")
        print(
            f"[crossearth] freeze_backbone={self.freeze_backbone_flag}, "
            f"train_patch_embed={self.train_patch_embed}, "
            f"in_channels={self.in_channels}"
        )

        return self

    # ------------------------------------------------------------------
    # Backbone setup
    # ------------------------------------------------------------------

    def _maybe_replace_patch_embed(self) -> None:
        """
        Replaces DINOv2 patch embedding if the requested input channels
        are different from the original DINOv2 RGB input.
        """
        if self.backbone is None:
            raise RuntimeError("Backbone non caricato.")

        if self.in_channels == 3:
            return

        print(
            f"[crossearth] Replacing patch embedding: "
            f"3 -> {self.in_channels} channels, "
            f"init='{self.patch_embed_init}'"
        )

        self.backbone = replace_patch_embed_input_channels(
            backbone=self.backbone,
            in_channels=self.in_channels,
            init_mode=self.patch_embed_init,
        )

    def _set_backbone_trainability(self) -> None:
        """
        Freezes or unfreezes the backbone.

        Recommended RGBNIR setup:
            - freeze the transformer backbone;
            - keep patch embedding trainable;
            - train Rein and decoder outside this module.
        """
        if self.backbone is None:
            return

        # First set all backbone parameters.
        for p in self.backbone.parameters():
            p.requires_grad = not self.freeze_backbone_flag

        # Optionally keep patch embedding trainable.
        if self.train_patch_embed and hasattr(self.backbone, "patch_embed"):
            for p in self.backbone.patch_embed.parameters():
                p.requires_grad = True

        if self.freeze_backbone_flag:
            self.backbone.eval()
        else:
            self.backbone.train()

    def set_freeze_backbone(self, freeze: bool = True) -> None:
        """
        Public helper to freeze/unfreeze the DINOv2 backbone after creation.
        """
        self.freeze_backbone_flag = freeze
        self._set_backbone_trainability()

    def unfreeze_last_blocks(self, n_blocks: int = 4) -> None:
        """
        Unfreezes only the last n transformer blocks of DINOv2.

        Useful for partial fine-tuning after a stable frozen-backbone run.

        Example:
            model.backbone_rein.unfreeze_last_blocks(n_blocks=4)
        """
        if self.backbone is None:
            raise RuntimeError("Backbone non caricato.")

        if n_blocks <= 0:
            return

        # Freeze everything first.
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Keep patch embedding trainable if requested.
        if self.train_patch_embed and hasattr(self.backbone, "patch_embed"):
            for p in self.backbone.patch_embed.parameters():
                p.requires_grad = True

        # Unfreeze last transformer blocks.
        blocks = self.backbone.blocks
        n_blocks = min(n_blocks, len(blocks))

        for block in blocks[-n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True

        self.backbone.train()

        print(f"[crossearth] Unfrozen last {n_blocks} DINOv2 blocks.")

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        """
        Registers forward hooks on selected DINOv2 transformer blocks.
        """
        if self.backbone is None:
            raise RuntimeError("Backbone not loaded. Cannot register hooks.")

        for h in self._hooks:
            h.remove()

        self._hooks.clear()

        for idx in self.out_indices:
            hook = self.backbone.blocks[idx].register_forward_hook(
                self._make_hook(idx)
            )
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            """
            DINOv2 output token layout:

                standard:
                    (B, 1 + N, D)

                with registers:
                    (B, 1 + R + N, D)

            where:
                1 = cls token
                R = register tokens
                N = patch tokens
            """
            out = self.rein.forward_layer(output, layer_idx)
            self._features.append((layer_idx, out))

        return hook

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> list:
        """
        Parameters
        ----------
        x:
            Input image tensor.

            RGB:
                shape = (B, 3, H, W)

            RGBNIR:
                shape = (B, 4, H, W)

        Requirements
        ------------
        H and W must be multiples of patch_size, usually 14.

        Returns
        -------
        out:
            List of feature maps:

                [F1, F2, F3, F4]

            Each feature has shape:

                (B, embed_dim, H / patch_size, W / patch_size)
        """
        if self.backbone is None:
            raise RuntimeError(
                "Backbone not loaded. Call load_backbone() before forward."
            )

        if x.ndim != 4:
            raise ValueError(
                f"x deve avere shape (B, C, H, W), ricevuto shape={x.shape}"
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Canali input incoerenti: modello creato con "
                f"in_channels={self.in_channels}, ma x ha C={x.shape[1]}."
            )

        img_h, img_w = x.shape[-2:]

        if img_h % self.patch_size != 0 or img_w % self.patch_size != 0:
            raise ValueError(
                f"H e W devono essere multipli di {self.patch_size}. "
                f"Ricevuto H={img_h}, W={img_w}."
            )

        h = img_h // self.patch_size
        w = img_w // self.patch_size
        expected_tokens = h * w

        self._features.clear()

        # If the full backbone is frozen and patch embedding is not trainable,
        # the entire backbone forward can be executed under no_grad().
        #
        # If patch embedding is trainable, do NOT use no_grad(), otherwise
        # gradients would not flow to the patch embedding.
        use_no_grad = self.freeze_backbone_flag and not self.train_patch_embed

        context = torch.no_grad() if use_no_grad else nullcontext()

        with context:
            _ = self.backbone(x)

        if len(self._features) != len(self.out_indices):
            raise RuntimeError(
                f"Numero feature catturate non valido: "
                f"{len(self._features)} invece di {len(self.out_indices)}. "
                "Controlla hooks e out_indices."
            )

        # Sort by layer index and reshape tokens to 2D feature maps.
        self._features.sort(key=lambda t: t[0])

        out = []

        for _, feat in self._features:
            # feat can be:
            #   (B, 1 + N, D)
            # or:
            #   (B, 1 + R + N, D)
            #
            # Remove cls token first.
            tokens = feat[:, 1:, :]

            actual_tokens = tokens.shape[1]

            if actual_tokens != expected_tokens:
                extra_tokens = actual_tokens - expected_tokens

                if extra_tokens > 0:
                    # DINOv2 *_reg models contain register tokens before patch tokens.
                    # Remove them and keep only patch tokens.
                    tokens = tokens[:, extra_tokens:, :]
                else:
                    raise RuntimeError(
                        f"Troppi pochi patch tokens: actual={actual_tokens}, "
                        f"expected={expected_tokens}"
                    )

            B, N, D = tokens.shape

            if N != expected_tokens:
                raise RuntimeError(
                    f"Numero patch tokens non coerente dopo rimozione register: "
                    f"N={N}, expected={expected_tokens}"
                )

            feat_2d = tokens.transpose(1, 2).reshape(B, D, h, w)
            out.append(feat_2d)

        self._features.clear()

        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def count_trainable_parameters(self) -> dict:
        """
        Returns a diagnostic count of trainable parameters inside this module.
        """
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        result = {
            "rein_trainable": count(self.rein),
        }

        if self.backbone is not None:
            result["backbone_trainable"] = count(self.backbone)

            if hasattr(self.backbone, "patch_embed"):
                result["patch_embed_trainable"] = count(self.backbone.patch_embed)

        return result