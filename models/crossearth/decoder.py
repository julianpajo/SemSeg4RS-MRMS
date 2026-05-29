"""
Pure PyTorch Mask2Former-style decoder
---------------------------------------------

This module provides:

    Mask2FormerDecoder
        - inference forward(features) -> dense logits (B, C, H, W)
        - training forward_train(features) -> list of class/mask predictions

Expected backbone features:
    features = [F1, F2, F3, F4]
    each F_i has shape (B, embed_dim, h_i, w_i)

For your CrossEarth/DINOv2 setup:
    h = H / patch_size
    w = W / patch_size
    patch_size = 14 for DINOv2 torch.hub variants

Changes vs original
-------------------
1. Pixel decoder: replaced flat sum with proper top-down FPN lateral connections.
2. num_queries: reduced from 100 to 20 for binary segmentation.
3. Self-attention in MaskedAttentionDecoderLayer: clarified variable naming
   (was accidentally computing key=q twice, now explicit query+pos for q/k,
   query only for v).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union
from typing import TypedDict


class Mask2FormerOutput(TypedDict):
    all_cls_preds: List[torch.Tensor]   # one tensor per decoder layer
    all_mask_preds: List[torch.Tensor]  # one tensor per decoder layer
    pred_logits: torch.Tensor           # (B, Q, num_classes+1)  — last layer
    pred_masks: torch.Tensor            # (B, Q, h, w)           — last layer

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Point sampling utilities
# =============================================================================

def point_sample(
    input: torch.Tensor,
    point_coords: torch.Tensor,
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Bilinear point sampling.

    Parameters
    ----------
    input:
        Tensor of shape (N, C, H, W).

    point_coords:
        Tensor of shape (N, P, 2) with coordinates normalized in [0, 1],
        ordered as (x, y).

    Returns
    -------
    output:
        Tensor of shape (N, C, P).
    """
    if point_coords.ndim != 3:
        raise ValueError(
            f"point_coords must have shape (N, P, 2), got {point_coords.shape}"
        )

    grid = point_coords * 2.0 - 1.0
    grid = grid.unsqueeze(2)  # (N, P, 1, 2)

    output = F.grid_sample(
        input,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=align_corners,
    )

    return output.squeeze(-1)  # (N, C, P)


@torch.no_grad()
def get_uncertain_point_coords_with_randomness(
    mask_logits: torch.Tensor,
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
) -> torch.Tensor:
    """
    Uncertainty-based point sampling from Mask2Former/PointRend.

    Parameters
    ----------
    mask_logits:
        Tensor of shape (N, 1, H, W), raw logits.

    num_points:
        Final number of sampled points.

    oversample_ratio:
        Candidate multiplier.

    importance_sample_ratio:
        Fraction of final points selected from most uncertain candidates.

    Returns
    -------
    point_coords:
        Tensor of shape (N, num_points, 2), normalized [0, 1].
    """
    if mask_logits.ndim != 4 or mask_logits.shape[1] != 1:
        raise ValueError(
            f"mask_logits must have shape (N, 1, H, W), got {mask_logits.shape}"
        )

    if oversample_ratio < 1.0:
        raise ValueError("oversample_ratio must be >= 1.0")

    if not (0.0 <= importance_sample_ratio <= 1.0):
        raise ValueError("importance_sample_ratio must be in [0, 1]")

    device = mask_logits.device
    n = mask_logits.shape[0]

    num_sampled = int(num_points * oversample_ratio)
    num_uncertain_points = int(num_points * importance_sample_ratio)
    num_random_points = num_points - num_uncertain_points

    candidate_coords = torch.rand(n, num_sampled, 2, device=device)

    candidate_logits = point_sample(
        mask_logits,
        candidate_coords,
        align_corners=False,
    ).squeeze(1)  # (N, num_sampled)

    uncertainties = -candidate_logits.abs()

    if num_uncertain_points > 0:
        idx = uncertainties.topk(num_uncertain_points, dim=1).indices
        uncertain_coords = torch.gather(
            candidate_coords,
            dim=1,
            index=idx.unsqueeze(-1).expand(-1, -1, 2),
        )
    else:
        uncertain_coords = candidate_coords[:, :0, :]

    if num_random_points > 0:
        random_coords = torch.rand(n, num_random_points, 2, device=device)
        point_coords = torch.cat([uncertain_coords, random_coords], dim=1)
    else:
        point_coords = uncertain_coords

    return point_coords


# =============================================================================
# Positional encoding
# =============================================================================

class SinePositionalEncoding2D(nn.Module):
    """
    Standard 2-D sine/cosine positional encoding.

    Output channels: 2 * num_feats
    """

    def __init__(
        self,
        num_feats: int = 128,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale or 2.0 * math.pi
        self.eps = eps

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        mask:
            Bool tensor of shape (B, H, W).
            True indicates padded/invalid positions.

        Returns
        -------
        pos:
            Tensor of shape (B, 2*num_feats, H, W).
        """
        if mask.ndim != 3:
            raise ValueError(f"mask must have shape (B, H, W), got {mask.shape}")

        not_mask = ~mask

        y_embed = not_mask.cumsum(dim=1, dtype=torch.float32)
        x_embed = not_mask.cumsum(dim=2, dtype=torch.float32)

        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = torch.arange(
            self.num_feats,
            dtype=torch.float32,
            device=mask.device,
        )
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
            dim=4,
        ).flatten(3)

        pos_y = torch.stack(
            (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
            dim=4,
        ).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3)
        pos = pos.permute(0, 3, 1, 2).contiguous()

        return pos


# =============================================================================
# Pixel decoder
# =============================================================================

class SimpleMask2FormerPixelDecoder(nn.Module):
    """
    FPN-style pixel decoder with proper top-down lateral connections.

    Input:
        4 feature maps from DINOv2/Rein at different resolutions.
        features[0] = lowest resolution (deepest), features[-1] = highest.

    Output:
        mask_features:
            High-resolution pixel embeddings (B, out_channels, h*4, w*4).

        multi_scale_memory:
            3 feature levels for the transformer decoder:
                level 0: (B, feat_channels, h,   w)
                level 1: (B, feat_channels, 2h,  2w)
                level 2: (B, feat_channels, 4h,  4w)

    Changes vs original
    -------------------
    Replaced flat sum-of-projected-features with a proper top-down FPN:
    features are projected per-level then fused via top-down addition with
    lateral connections, preserving the spatial hierarchy (deep features
    provide global semantics, shallow features provide local detail).
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        feat_channels: int = 256,
        out_channels: int = 256,
        num_input_levels: int = 4,
        num_transformer_levels: int = 3,
        norm_groups: int = 32,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.feat_channels = feat_channels
        self.out_channels = out_channels
        self.num_input_levels = num_input_levels
        self.num_transformer_levels = num_transformer_levels

        # Per-level lateral projections: embed_dim -> feat_channels
        self.input_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(embed_dim, feat_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(norm_groups, feat_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_input_levels)
            ]
        )

        # Per-level output convolutions applied after top-down fusion
        self.output_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(norm_groups, feat_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_input_levels)
            ]
        )

        # Projections to produce the num_transformer_levels memory tensors
        self.level_projs = nn.ModuleList(
            [
                nn.Conv2d(feat_channels, feat_channels, kernel_size=1)
                for _ in range(num_transformer_levels)
            ]
        )

        self.mask_feature = nn.Conv2d(feat_channels, out_channels, kernel_size=1)

    def forward(
        self,
        features: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Parameters
        ----------
        features:
            List of num_input_levels tensors, each (B, embed_dim, h_i, w_i).
            Expected order: coarse-to-fine, i.e.
                features[0]  = (B, D, h,    w)     <- lowest res / deepest
                features[-1] = (B, D, h*8,  w*8)   <- highest res / shallowest

        Returns
        -------
        mask_features:
            (B, out_channels, h_finest, w_finest)

        multi_scale_memory:
            List of num_transformer_levels tensors (coarse to fine).
        """
        if len(features) != self.num_input_levels:
            raise ValueError(
                f"Expected {self.num_input_levels} features, got {len(features)}"
            )

        # Step 1: lateral projections
        projected = [proj(feat) for feat, proj in zip(features, self.input_projs)]

        # Step 2: top-down FPN fusion (from coarsest to finest)
        # projected[0] is the coarsest level; we propagate downward.
        fpn_outs = [None] * self.num_input_levels
        fpn_outs[0] = self.output_convs[0](projected[0])

        for i in range(1, self.num_input_levels):
            target_size = projected[i].shape[-2:]
            top_down = F.interpolate(
                fpn_outs[i - 1],
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            fpn_outs[i] = self.output_convs[i](projected[i] + top_down)

        # Step 3: select num_transformer_levels memory tensors (coarse to fine)
        # We take the last num_transformer_levels FPN outputs.
        selected = fpn_outs[-self.num_transformer_levels:]  # fine levels

        multi_scale_memory = [
            proj(feat) for feat, proj in zip(selected, self.level_projs)
        ]

        # Step 4: mask features from the finest FPN level
        mask_features = self.mask_feature(fpn_outs[-1])

        return mask_features, multi_scale_memory


# =============================================================================
# Transformer decoder
# =============================================================================

class MaskedAttentionDecoderLayer(nn.Module):
    """
    Mask2Former decoder layer:

        masked cross-attention -> self-attention -> FFN

    Changes vs original
    -------------------
    Clarified self-attention: query and key receive query_pos added,
    value receives query only (no positional embedding), which matches
    the standard Mask2Former / DETR convention.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        query:      (B, Q, D)
        key/value:  (B, HW, D)
        query_pos:  (B, Q, D)
        key_pos:    (B, HW, D)
        attn_mask:  Bool tensor (B*num_heads, Q, HW). True = blocked.
        """
        # --- Masked cross-attention ---
        q = query + query_pos
        k = key + key_pos

        query2, _ = self.cross_attn(
            query=q,
            key=k,
            value=value,          # value has no positional embedding
            attn_mask=attn_mask,
            need_weights=False,
        )
        query = self.norm1(query + query2)

        # --- Self-attention ---
        # query_pos added to q and k only; v = query (no pos embedding)
        q = query + query_pos
        query2, _ = self.self_attn(
            query=q,
            key=q,                # same as query + query_pos
            value=query,          # explicit: no positional embedding on value
            need_weights=False,
        )
        query = self.norm2(query + query2)

        # --- FFN ---
        query = self.norm3(query + self.ffn(query))

        return query


class MLP(nn.Module):
    """Simple MLP used for mask embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers = []

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim

            layers.append(nn.Linear(in_dim, out_dim))

            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Mask2Former decoder
# =============================================================================

class Mask2FormerDecoder(nn.Module):
    """
    Pure PyTorch Mask2Former-style decoder for DINOv2/Rein features.

    Inference:
        logits = decoder(features)          # (B, num_classes, H, W)

    Training:
        outputs = decoder.forward_train(features)
        loss    = criterion(outputs, targets)

    Parameters
    ----------
    num_queries : int
        Number of object queries. Default is 20, suitable for binary
        segmentation. Use larger values (e.g. 100) for many-class datasets.

    Notes
    -----
    forward() returns dense class score maps suitable for argmax/eval.
    For training with Mask2Former losses, use forward_train().
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_classes: int = 2,
        feat_channels: int = 256,
        out_channels: int = 256,
        num_queries: int = 20,          # reduced from 100: sufficient for binary seg
        num_decoder_layers: int = 9,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        patch_size: int = 14,
        num_input_levels: int = 4,
        num_transformer_feat_level: int = 3,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.out_channels = out_channels
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_transformer_feat_level = num_transformer_feat_level

        self.pixel_decoder = SimpleMask2FormerPixelDecoder(
            embed_dim=embed_dim,
            feat_channels=feat_channels,
            out_channels=out_channels,
            num_input_levels=num_input_levels,
            num_transformer_levels=num_transformer_feat_level,
        )

        self.decoder_input_projs = nn.ModuleList(
            [
                nn.Identity()
                for _ in range(num_transformer_feat_level)
            ]
        )

        self.positional_encoding = SinePositionalEncoding2D(
            num_feats=feat_channels // 2,
            normalize=True,
        )

        self.query_feat = nn.Embedding(num_queries, feat_channels)
        self.query_embed = nn.Embedding(num_queries, feat_channels)
        self.level_embed = nn.Embedding(num_transformer_feat_level, feat_channels)

        self.layers = nn.ModuleList(
            [
                MaskedAttentionDecoderLayer(
                    d_model=feat_channels,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.decoder_norm = nn.LayerNorm(feat_channels)

        # +1 for the no-object class
        self.cls_embed = nn.Linear(feat_channels, num_classes + 1)

        self.mask_embed = MLP(
            input_dim=feat_channels,
            hidden_dim=feat_channels,
            output_dim=out_channels,
            num_layers=3,
        )

    def _prepare_decoder_inputs(
        self,
        multi_scale_memory: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Tuple[int, int]]]:
        """Converts multi-scale feature maps to flattened transformer memories."""
        decoder_inputs = []
        decoder_pos = []
        spatial_shapes = []

        for level_idx, x in enumerate(multi_scale_memory):
            if level_idx >= self.num_transformer_feat_level:
                break

            x = self.decoder_input_projs[level_idx](x)

            b, c, h, w = x.shape
            spatial_shapes.append((h, w))

            x_flat = x.flatten(2).permute(0, 2, 1).contiguous()
            x_flat = x_flat + self.level_embed.weight[level_idx].view(1, 1, -1)

            padding_mask = torch.zeros(b, h, w, dtype=torch.bool, device=x.device)
            pos = self.positional_encoding(padding_mask)
            pos_flat = pos.flatten(2).permute(0, 2, 1).contiguous()

            decoder_inputs.append(x_flat)
            decoder_pos.append(pos_flat)

        return decoder_inputs, decoder_pos, spatial_shapes

    def _prediction_heads(
        self,
        query_feat: torch.Tensor,
        mask_features: torch.Tensor,
        attn_mask_target_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produces class logits, mask logits, and attention mask."""
        decoder_output = self.decoder_norm(query_feat)

        cls_pred = self.cls_embed(decoder_output)       # (B, Q, num_classes+1)

        mask_embed = self.mask_embed(decoder_output)
        mask_pred = torch.einsum(
            "bqd,bdhw->bqhw",
            mask_embed,
            mask_features,
        )                                               # (B, Q, h_mask, w_mask)

        attn_mask = F.interpolate(
            mask_pred,
            size=attn_mask_target_size,
            mode="bilinear",
            align_corners=False,
        )

        attn_mask = attn_mask.flatten(2)                # (B, Q, HW)
        attn_mask = (
            attn_mask
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
        )                                               # (B*num_heads, Q, HW)

        attn_mask = (attn_mask.sigmoid() < 0.5).detach()   # True = blocked

        return cls_pred, mask_pred, attn_mask

    def _run_decoder(
        self,
        mask_features: torch.Tensor,
        decoder_inputs: List[torch.Tensor],
        decoder_pos: List[torch.Tensor],
        spatial_shapes: List[Tuple[int, int]],
    ) -> Mask2FormerOutput:
        """Runs transformer decoder and returns predictions from all layers."""
        b = mask_features.shape[0]

        query_feat = self.query_feat.weight.unsqueeze(0).repeat(b, 1, 1)
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)

        all_cls_preds = []
        all_mask_preds = []

        # Initial prediction before any decoder layer
        cls_pred, mask_pred, attn_mask = self._prediction_heads(
            query_feat=query_feat,
            mask_features=mask_features,
            attn_mask_target_size=spatial_shapes[0],
        )
        all_cls_preds.append(cls_pred)
        all_mask_preds.append(mask_pred)

        for layer_idx, layer in enumerate(self.layers):
            level_idx = layer_idx % self.num_transformer_feat_level

            # Avoid NaNs when an entire row is masked
            if attn_mask is not None:
                all_true = attn_mask.sum(dim=-1) == attn_mask.shape[-1]
                attn_mask[all_true] = False

            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_pos[level_idx],
                attn_mask=attn_mask,
            )

            next_level_idx = (layer_idx + 1) % self.num_transformer_feat_level

            cls_pred, mask_pred, attn_mask = self._prediction_heads(
                query_feat=query_feat,
                mask_features=mask_features,
                attn_mask_target_size=spatial_shapes[next_level_idx],
            )
            all_cls_preds.append(cls_pred)
            all_mask_preds.append(mask_pred)

        return {
            "all_cls_preds": all_cls_preds,
            "all_mask_preds": all_mask_preds,
            "pred_logits": all_cls_preds[-1],
            "pred_masks": all_mask_preds[-1],
        }

    def forward_train(self, features: List[torch.Tensor]) -> Mask2FormerOutput:
        """Returns raw Mask2Former predictions for loss computation."""
        mask_features, multi_scale_memory = self.pixel_decoder(features)
        decoder_inputs, decoder_pos, spatial_shapes = self._prepare_decoder_inputs(
            multi_scale_memory
        )
        return self._run_decoder(
            mask_features=mask_features,
            decoder_inputs=decoder_inputs,
            decoder_pos=decoder_pos,
            spatial_shapes=spatial_shapes,
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Inference forward.

        Returns
        -------
        logits:
            Dense segmentation scores, shape (B, num_classes, H, W).
        """
        outputs = self.forward_train(features)

        cls_pred = outputs["pred_logits"]   # (B, Q, num_classes+1)
        mask_pred = outputs["pred_masks"]   # (B, Q, h_mask, w_mask)

        # Softmax over num_classes+1, then drop the no-object column.
        # This correctly normalises over all classes including background.
        cls_prob = cls_pred.softmax(dim=-1)[..., :self.num_classes]  # (B, Q, C)

        mask_prob = mask_pred.sigmoid()     # (B, Q, h_mask, w_mask)

        logits = torch.einsum(
            "bqc,bqhw->bchw",
            cls_prob,
            mask_prob,
        )                                   # (B, C, h_mask, w_mask)

        # Upsample to original image resolution
        out_h = features[0].shape[-2] * self.patch_size
        out_w = features[0].shape[-1] * self.patch_size

        logits = F.interpolate(
            logits,
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
        )

        return logits