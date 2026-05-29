"""
pl_modules/components/losses.py
--------------------------------
Loss components for semantic segmentation training.

Each loss is implemented as a configuration class that inherits from
``Loss`` and exposes a ``build()`` method returning the corresponding
``nn.Module``. This pattern decouples hyperparameter configuration from
instantiation and integrates cleanly with ``LightningSegConfig``.

Available components
--------------------
CrossEntropyLoss
    Standard cross-entropy with optional per-class weights and label
    smoothing. Suitable for most segmentation tasks.

ImbalanceLoss
    Boundary-aware imbalance loss that combines weighted cross-entropy
    with a boundary term penalising misclassified edge pixels. Designed
    for class-imbalanced datasets such as coastal wetlands.

Mask2FormerLoss
    Hungarian-matched loss combining classification cross-entropy,
    point-sampled sigmoid BCE, and Dice loss across all decoder layers.
    Used for instance-aware segmentation heads.

Mask2FormerLossConfig
    Configuration wrapper around ``Mask2FormerLoss`` compatible with the
    ``Loss.build()`` interface.

Requirements
------------
- torch
- scipy  (required by ``Mask2FormerLoss`` for Hungarian matching)

Usage
-----
    from pl_modules.components.losses import CrossEntropyLoss, ImbalanceLoss

    # Standard CE
    criterion = CrossEntropyLoss(ignore_index=255).build()
    loss = criterion(logits, target)

    # Imbalance loss with class weights
    criterion = ImbalanceLoss(
        num_classes=2,
        ignore_index=255,
        alpha=0.7,
        class_weights=[1.0, 2.5],
    ).build()
    loss = criterion(logits, target)

    # Mask2Former loss
    from pl_modules.components.losses import Mask2FormerLossConfig
    criterion = Mask2FormerLossConfig(num_classes=2, ignore_index=255).build()
    losses = criterion(outputs, target)
    loss = losses["loss"]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Loss

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:
    raise ImportError(
        "Mask2FormerLoss requires scipy. Install with: pip install scipy"
    ) from exc


# ============================================================
# CrossEntropy
# ============================================================

class CrossEntropyLoss(Loss):
    """
    Standard CrossEntropyLoss.

    Parameters
    ----------
    ignore_index  : int
        Pixels to ignore.

    class_weights : list | None
        Per-class weights.

    label_smoothing : float
        Label smoothing factor.
    """

    def __init__(
        self,
        ignore_index: int = 255,
        class_weights: Optional[List[float]] = None,
        label_smoothing: float = 0.0,
    ):
        self.ignore_index = ignore_index
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def build(self) -> nn.Module:
        weight = (
            torch.tensor(self.class_weights, dtype=torch.float32)
            if self.class_weights is not None
            else None
        )

        return nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )


# ============================================================
# IL — Imbalance Loss
# ============================================================

class _IL(nn.Module):
    """Internal IL module returned by ImbalanceLoss.build()."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int,
        alpha: float,
        class_weights: Optional[List[float]],
        boundary_kernel: int,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.boundary_kernel = boundary_kernel

        weight = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )

        self.ce_weighted = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction="mean",
        )

        self.ce_boundary = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction="none",
        )

    def _boundary_mask(self, target: torch.Tensor) -> torch.Tensor:
        t = target.clone().float()
        t[t == self.ignore_index] = -1.0

        k = self.boundary_kernel
        pad = k // 2
        t4 = t.unsqueeze(1)

        t_max = F.max_pool2d(t4, kernel_size=k, stride=1, padding=pad)
        t_min = -F.max_pool2d(-t4, kernel_size=k, stride=1, padding=pad)

        boundary = (t_max - t_min > 0).squeeze(1)
        valid = target != self.ignore_index

        return (boundary & valid).float()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_wce = self.ce_weighted(logits, target)
        boundary = self._boundary_mask(target)

        if boundary.sum() > 0:
            l_pixel = self.ce_boundary(logits, target)
            l_boundary = (l_pixel * boundary).sum() / boundary.sum()
        else:
            l_boundary = l_wce.new_tensor(0.0)

        return self.alpha * l_wce + (1.0 - self.alpha) * l_boundary


class ImbalanceLoss(Loss):
    """
    Imbalance Loss.

    Combines weighted CrossEntropy with a boundary-aware term.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        alpha: float = 0.7,
        class_weights: Optional[List[float]] = None,
        boundary_kernel: int = 3,
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.class_weights = class_weights
        self.boundary_kernel = boundary_kernel

    def build(self) -> nn.Module:
        return _IL(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            alpha=self.alpha,
            class_weights=self.class_weights,
            boundary_kernel=self.boundary_kernel,
        )


# ============================================================
# Mask2Former utilities
# ============================================================

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
        Tensor of shape (N, P, 2), normalized in [0, 1],
        ordered as (x, y).

    Returns
    -------
    Tensor of shape (N, C, P).
    """
    if point_coords.ndim != 3:
        raise ValueError(
            f"point_coords must have shape (N, P, 2), got {point_coords.shape}"
        )

    grid = point_coords * 2.0 - 1.0
    grid = grid.unsqueeze(2)

    output = F.grid_sample(
        input,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=align_corners,
    )

    return output.squeeze(-1)


@torch.no_grad()
def get_uncertain_point_coords_with_randomness(
    mask_logits: torch.Tensor,
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
) -> torch.Tensor:
    """
    Uncertainty-based point sampling used by Mask2Former.

    Uncertainty is highest when the mask logit is close to zero.
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
    ).squeeze(1)

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


# ============================================================
# Mask2Former pairwise matching losses
# ============================================================

def batch_sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Pairwise sigmoid BCE loss.

    Parameters
    ----------
    inputs:
        Predicted mask logits, shape (Q, P).

    targets:
        Target masks, shape (T, P).

    Returns
    -------
    Pairwise loss matrix, shape (Q, T).
    """
    q, p = inputs.shape
    t = targets.shape[0]

    if t == 0:
        return inputs.new_zeros((q, 0))

    inputs = inputs.float()
    targets = targets.float()

    pos = F.binary_cross_entropy_with_logits(
        inputs,
        torch.ones_like(inputs),
        reduction="none",
    )

    neg = F.binary_cross_entropy_with_logits(
        inputs,
        torch.zeros_like(inputs),
        reduction="none",
    )

    loss = torch.einsum("qp,tp->qt", pos, targets)
    loss = loss + torch.einsum("qp,tp->qt", neg, 1.0 - targets)

    return loss / p


def batch_dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1.0,
) -> torch.Tensor:
    """
    Pairwise Dice loss.

    Parameters
    ----------
    inputs:
        Predicted mask logits, shape (Q, P).

    targets:
        Target masks, shape (T, P).

    Returns
    -------
    Pairwise loss matrix, shape (Q, T).
    """
    q, _ = inputs.shape
    t = targets.shape[0]

    if t == 0:
        return inputs.new_zeros((q, 0))

    inputs = inputs.sigmoid()
    targets = targets.float()

    numerator = 2.0 * torch.einsum("qp,tp->qt", inputs, targets)
    denominator = inputs.sum(dim=1)[:, None] + targets.sum(dim=1)[None, :]

    return 1.0 - (numerator + eps) / (denominator + eps)


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
) -> torch.Tensor:
    """
    Sigmoid BCE loss for matched masks.

    inputs/targets:
        shape (N, P)
    """
    loss = F.binary_cross_entropy_with_logits(
        inputs,
        targets,
        reduction="none",
    )

    return loss.mean(dim=1).sum() / max(num_masks, 1.0)


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    eps: float = 1.0,
) -> torch.Tensor:
    """
    Dice loss for matched masks.

    inputs/targets:
        shape (N, P)
    """
    inputs = inputs.sigmoid()

    numerator = 2.0 * (inputs * targets).sum(dim=1)
    denominator = inputs.sum(dim=1) + targets.sum(dim=1)

    loss = 1.0 - (numerator + eps) / (denominator + eps)

    return loss.sum() / max(num_masks, 1.0)


# ============================================================
# Mask2Former output type
# ============================================================

# Imported by the decoder too — single source of truth.
# all_cls_preds / all_mask_preds : one tensor per decoder layer (aux + final)
# pred_logits / pred_masks       : final-layer tensors (convenience aliases)
Mask2FormerOutput = Dict[str, Any]


# ============================================================
# Mask2Former criterion
# ============================================================

class Mask2FormerLoss(nn.Module):
    """
    Mask2Former loss with Hungarian matching.

    Expected outputs (Mask2FormerOutput):
        outputs["all_cls_preds"]  : List[Tensor(B, Q, C+1)]
        outputs["all_mask_preds"] : List[Tensor(B, Q, Hm, Wm)]
        outputs["pred_logits"]    : Tensor(B, Q, C+1)   — last layer alias
        outputs["pred_masks"]     : Tensor(B, Q, Hm, Wm) — last layer alias

    Expected targets:
        Dense semantic labels : LongTensor (B, H, W)
        or list of dicts      : [{"labels": LongTensor(N_i), "masks": FloatTensor(N_i,H,W)}, ...]

    Loss weights
    ------------
    loss_cls_weight   : weight on classification CE (applied to both final and aux)
    loss_mask_weight  : weight on point-sampled BCE
    loss_dice_weight  : weight on point-sampled Dice
    aux_loss_weight   : additional scalar multiplier for all auxiliary layer losses
                        (final layer is not affected)

    Note: loss_cls_weight / loss_mask_weight / loss_dice_weight are also used
    in the Hungarian matching cost matrix — this is intentional and matches
    the original Mask2Former implementation.

    Classification loss normalisation
    ----------------------------------
    Following the paper, the CE loss is also divided by num_masks so that
    all three loss terms are on a comparable scale regardless of batch size.
    """

    _class_weight: Optional[torch.Tensor]

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        loss_cls_weight: float = 2.0,
        loss_mask_weight: float = 5.0,
        loss_dice_weight: float = 5.0,
        bg_cls_weight: float = 0.1,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        aux_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.loss_cls_weight = loss_cls_weight
        self.loss_mask_weight = loss_mask_weight
        self.loss_dice_weight = loss_dice_weight
        self.bg_cls_weight = bg_cls_weight

        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.aux_loss_weight = aux_loss_weight

        class_weight = torch.ones(num_classes + 1, dtype=torch.float32)
        class_weight[-1] = bg_cls_weight

        self.register_buffer("_class_weight", class_weight)

    @property
    def class_weight(self) -> torch.Tensor:
        return cast(torch.Tensor, self._class_weight)

    def _semantic_targets_to_instances(
        self,
        semantic_targets: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Converts dense semantic labels to per-class binary masks.
        """
        if semantic_targets.ndim != 3:
            raise ValueError(
                f"semantic_targets must have shape (B, H, W), "
                f"got {semantic_targets.shape}"
            )

        targets = []

        for target in semantic_targets:
            valid = target != self.ignore_index
            labels = []

            for class_id in range(self.num_classes):
                if torch.any((target == class_id) & valid):
                    labels.append(class_id)

            if len(labels) == 0:
                labels_tensor = torch.empty(
                    0,
                    dtype=torch.long,
                    device=semantic_targets.device,
                )
                masks_tensor = torch.empty(
                    0,
                    target.shape[-2],
                    target.shape[-1],
                    dtype=torch.float32,
                    device=semantic_targets.device,
                )
            else:
                labels_tensor = torch.tensor(
                    labels,
                    dtype=torch.long,
                    device=semantic_targets.device,
                )

                masks_tensor = torch.stack(
                    [
                        ((target == class_id) & valid).float()
                        for class_id in labels
                    ],
                    dim=0,
                )

            targets.append(
                {
                    "labels": labels_tensor,
                    "masks": masks_tensor,
                }
            )

        return targets

    def _normalize_targets(
        self,
        targets: Union[torch.Tensor, List[Dict[str, torch.Tensor]]],
        device: torch.device,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Accepts either dense semantic labels or list-of-dicts targets.
        """
        if isinstance(targets, torch.Tensor):
            return self._semantic_targets_to_instances(targets.to(device))

        if not isinstance(targets, list):
            raise TypeError(
                "targets must be either a Tensor (B,H,W) or a list of dictionaries"
            )

        normalized = []

        for target in targets:
            if "labels" not in target or "masks" not in target:
                raise KeyError("Each target dict must contain 'labels' and 'masks'.")

            labels = target["labels"].to(device=device, dtype=torch.long)
            masks = target["masks"].to(device=device, dtype=torch.float32)

            normalized.append(
                {
                    "labels": labels,
                    "masks": masks,
                }
            )

        return normalized

    @torch.no_grad()
    def _match_single(
        self,
        cls_pred: torch.Tensor,
        mask_pred: torch.Tensor,
        target: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hungarian matching for one image.
        """
        labels = target["labels"]
        masks = target["masks"]

        num_targets = labels.shape[0]

        if num_targets == 0:
            return (
                torch.empty(0, dtype=torch.long, device=cls_pred.device),
                torch.empty(0, dtype=torch.long, device=cls_pred.device),
            )

        # Classification cost
        cls_prob = cls_pred.softmax(dim=-1)
        cost_class = -cls_prob[:, labels]

        # Resize targets to predicted mask resolution
        masks = F.interpolate(
            masks[:, None],
            size=mask_pred.shape[-2:],
            mode="nearest",
        ).squeeze(1)

        # Random points for matching
        point_coords = torch.rand(
            1,
            self.num_points,
            2,
            device=mask_pred.device,
        )

        point_coords_pred = point_coords.repeat(mask_pred.shape[0], 1, 1)
        point_coords_tgt = point_coords.repeat(num_targets, 1, 1)

        pred_points = point_sample(
            mask_pred[:, None],
            point_coords_pred,
            align_corners=False,
        ).squeeze(1)

        tgt_points = point_sample(
            masks[:, None],
            point_coords_tgt,
            align_corners=False,
        ).squeeze(1)

        cost_mask = batch_sigmoid_ce_loss(pred_points, tgt_points)
        cost_dice = batch_dice_loss(pred_points, tgt_points)

        # Note: same weights as the loss — intentional (matches original paper).
        cost = (
            self.loss_cls_weight * cost_class
            + self.loss_mask_weight * cost_mask
            + self.loss_dice_weight * cost_dice
        )

        cost = cost.detach().cpu()

        src_idx_np, tgt_idx_np = linear_sum_assignment(cost)

        src_idx = torch.as_tensor(
            src_idx_np,
            dtype=torch.long,
            device=cls_pred.device,
        )
        tgt_idx = torch.as_tensor(
            tgt_idx_np,
            dtype=torch.long,
            device=cls_pred.device,
        )

        return src_idx, tgt_idx

    @torch.no_grad()
    def _match_batch(
        self,
        cls_preds: torch.Tensor,
        mask_preds: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Hungarian matching for the whole batch.
        """
        return [
            self._match_single(
                cls_pred=cls_preds[b],
                mask_pred=mask_preds[b],
                target=targets[b],
            )
            for b in range(cls_preds.shape[0])
        ]

    def _loss_labels(
        self,
        cls_preds: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_masks: float,
    ) -> torch.Tensor:
        """
        Classification CE loss.

        Unmatched queries are assigned to the no-object class (index num_classes).
        Normalised by num_masks to keep the same scale as mask and dice losses.
        """
        b, q, _ = cls_preds.shape

        target_classes = torch.full(
            (b, q),
            fill_value=self.num_classes,
            dtype=torch.long,
            device=cls_preds.device,
        )

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue

            target_classes[batch_idx, src_idx] = (
                targets[batch_idx]["labels"][tgt_idx]
            )

        loss_cls = F.cross_entropy(
            cls_preds.transpose(1, 2),
            target_classes,
            weight=self.class_weight,
        )

        # Normalise by num_masks (paper convention)
        return loss_cls / max(num_masks, 1.0)

    def _loss_masks(
        self,
        mask_preds: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_masks: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Point-sampled BCE and Dice losses for matched masks.
        """
        src_masks = []
        tgt_masks = []

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue

            src = mask_preds[batch_idx, src_idx]

            tgt = targets[batch_idx]["masks"][tgt_idx]
            tgt = F.interpolate(
                tgt[:, None],
                size=src.shape[-2:],
                mode="nearest",
            ).squeeze(1)

            src_masks.append(src)
            tgt_masks.append(tgt)

        if len(src_masks) == 0:
            zero = mask_preds.sum() * 0.0
            return zero, zero

        src_masks = torch.cat(src_masks, dim=0)
        tgt_masks = torch.cat(tgt_masks, dim=0)

        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                mask_logits=src_masks[:, None],
                num_points=self.num_points,
                oversample_ratio=self.oversample_ratio,
                importance_sample_ratio=self.importance_sample_ratio,
            )

            tgt_points = point_sample(
                tgt_masks[:, None],
                point_coords,
                align_corners=False,
            ).squeeze(1)

        src_points = point_sample(
            src_masks[:, None],
            point_coords,
            align_corners=False,
        ).squeeze(1)

        loss_mask = sigmoid_ce_loss(
            src_points,
            tgt_points,
            num_masks=num_masks,
        )

        loss_dice = dice_loss(
            src_points,
            tgt_points,
            num_masks=num_masks,
        )

        return loss_mask, loss_dice

    def _loss_single_layer(
        self,
        cls_preds: torch.Tensor,
        mask_preds: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Computes losses for one decoder layer.
        """
        indices = self._match_batch(
            cls_preds=cls_preds,
            mask_preds=mask_preds,
            targets=targets,
        )

        num_masks = float(max(sum(t["labels"].numel() for t in targets), 1))

        loss_cls = self._loss_labels(
            cls_preds=cls_preds,
            targets=targets,
            indices=indices,
            num_masks=num_masks,
        )

        loss_mask, loss_dice = self._loss_masks(
            mask_preds=mask_preds,
            targets=targets,
            indices=indices,
            num_masks=num_masks,
        )

        return {
            "loss_cls":  loss_cls  * self.loss_cls_weight,
            "loss_mask": loss_mask * self.loss_mask_weight,
            "loss_dice": loss_dice * self.loss_dice_weight,
        }

    def forward(
        self,
        outputs: Mask2FormerOutput,
        targets: Union[torch.Tensor, List[Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        outputs : Mask2FormerOutput
            Dictionary produced by Mask2FormerDecoder.forward_train().

        targets : Tensor(B,H,W) or List[Dict]
            Ground-truth labels (dense or instance-style).

        Returns
        -------
        Dictionary with keys:
            loss           — total scalar loss
            loss_cls       — final-layer classification loss
            loss_mask      — final-layer BCE mask loss
            loss_dice      — final-layer Dice loss
            loss_cls_aux_i   \\
            loss_mask_aux_i   > one entry per intermediate decoder layer i
            loss_dice_aux_i  /
        """
        pred_logits: torch.Tensor = outputs["pred_logits"]
        device = pred_logits.device

        normalized_targets = self._normalize_targets(targets, device=device)

        all_cls_preds: List[torch.Tensor] = outputs["all_cls_preds"]
        all_mask_preds: List[torch.Tensor] = outputs["all_mask_preds"]

        # Final-layer losses
        final_losses = self._loss_single_layer(
            cls_preds=all_cls_preds[-1],
            mask_preds=all_mask_preds[-1],
            targets=normalized_targets,
        )

        losses: Dict[str, torch.Tensor] = {
            "loss_cls":  final_losses["loss_cls"],
            "loss_mask": final_losses["loss_mask"],
            "loss_dice": final_losses["loss_dice"],
        }

        # Auxiliary losses for all intermediate decoder layers.
        # Each term is scaled by aux_loss_weight on top of the per-loss weights
        # already applied inside _loss_single_layer.
        for i, (cls_pred, mask_pred) in enumerate(
            zip(all_cls_preds[:-1], all_mask_preds[:-1])
        ):
            aux = self._loss_single_layer(
                cls_preds=cls_pred,
                mask_preds=mask_pred,
                targets=normalized_targets,
            )

            losses[f"loss_cls_aux_{i}"]  = aux["loss_cls"]  * self.aux_loss_weight
            losses[f"loss_mask_aux_{i}"] = aux["loss_mask"] * self.aux_loss_weight
            losses[f"loss_dice_aux_{i}"] = aux["loss_dice"] * self.aux_loss_weight

        losses["loss"] = torch.stack(list(losses.values())).sum()

        return losses


class Mask2FormerLossConfig(Loss):
    """
    Config wrapper compatible with the project Loss.build() interface.

    Example
    -------
    criterion_cfg = Mask2FormerLossConfig(
        num_classes=2,
        ignore_index=255,
    )

    criterion = criterion_cfg.build()
    losses = criterion(outputs, target)
    loss = losses["loss"]
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        loss_cls_weight: float = 2.0,
        loss_mask_weight: float = 5.0,
        loss_dice_weight: float = 5.0,
        bg_cls_weight: float = 0.1,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        aux_loss_weight: float = 1.0,
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.loss_cls_weight = loss_cls_weight
        self.loss_mask_weight = loss_mask_weight
        self.loss_dice_weight = loss_dice_weight
        self.bg_cls_weight = bg_cls_weight
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.aux_loss_weight = aux_loss_weight

    def build(self) -> nn.Module:
        return Mask2FormerLoss(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            loss_cls_weight=self.loss_cls_weight,
            loss_mask_weight=self.loss_mask_weight,
            loss_dice_weight=self.loss_dice_weight,
            bg_cls_weight=self.bg_cls_weight,
            num_points=self.num_points,
            oversample_ratio=self.oversample_ratio,
            importance_sample_ratio=self.importance_sample_ratio,
            aux_loss_weight=self.aux_loss_weight,
        )