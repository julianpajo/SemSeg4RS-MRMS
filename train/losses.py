"""
models/losses.py
----------------

Loss functions aligned with the original papers/configurations of each model.

  segformer_sae  -> WIL (Wetland Imbalance Loss)
                    = 0.5 × DiceLoss + 0.5 × FocalLoss  [Eq. 16-18, paper]
                    gamma=2, optional class weights for Focal

  DeepLabV3+     -> standard CrossEntropy  [Chen et al. 2018]
                    with optional class_weights inversely proportional to class frequency

  dofa           -> standard CrossEntropy  [official dofa-pytorch mmseg config]

  crossearth     -> standard CrossEntropy  [official crossearth/Rein mmseg config]

All losses support ignore_index=255.

Usage
-----
    from models.losses import build_loss

    criterion = build_loss("segformer_sae", num_classes=14, ignore_index=255)
    criterion = build_loss("deeplabv3plus", num_classes=14, class_weights=weights)
    criterion = build_loss("dofa")
    criterion = build_loss("crossearth")

    loss_dict = criterion(logits, targets)
    loss_dict["loss"].backward()
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dice Loss used in WIL
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Soft multi-class Macro-Dice Loss [Eq. 16 of the paper].

    Formula
    -------
        L_dice = 1 - (2 * TP) / (2 * TP + FP + FN)

    Dice is computed for each class and then averaged using macro averaging.
    Pixels equal to ignore_index are excluded from the computation.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255, eps: float = 1e-6):
        """
        Initialize the Dice loss.

        Parameters
        ----------
        num_classes :
            Number of semantic classes.
        ignore_index :
            Label value ignored during loss computation.
        eps :
            Numerical stability constant.
        """
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the Dice loss.

        Parameters
        ----------
        logits :
            Model logits with shape (B, C, H, W).
        targets :
            Ground-truth labels with shape (B, H, W).

        Returns
        -------
        torch.Tensor
            Scalar Dice loss.
        """
        probs = logits.softmax(dim=1)           # (B, C, H, W)

        valid = targets != self.ignore_index    # (B, H, W)
        t = targets.clone()
        t[~valid] = 0

        _, _, _, _ = probs.shape
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, t.unsqueeze(1), 1.0)

        mask = valid.unsqueeze(1).float()
        probs = probs * mask
        one_hot = one_hot * mask

        dims = (0, 2, 3)
        inter = (probs * one_hot).sum(dims)
        union = probs.sum(dims) + one_hot.sum(dims)
        dice = (2.0 * inter + self.eps) / (union + self.eps)

        return 1.0 - dice.mean()


# ---------------------------------------------------------------------------
# Focal Loss used in WIL
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss [Lin et al. 2017, Eq. 17 of the paper].

    Formula
    -------
        L_focal = -sum_c w_c sum_i (1 - p_{i,c})^gamma log(p_{i,c})

    gamma=2 is used as stated in the paper section 3.5:
    "gamma is set to 2".

    Class weights w_c are optional.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Initialize the Focal loss.

        Parameters
        ----------
        num_classes :
            Number of semantic classes.
        ignore_index :
            Label value ignored during loss computation.
        gamma :
            Focusing parameter.
        class_weights :
            Optional tensor with shape (C,) containing class weights.
        """
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None
            else torch.ones(num_classes)
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the Focal loss.

        Parameters
        ----------
        logits :
            Model logits with shape (B, C, H, W).
        targets :
            Ground-truth labels with shape (B, H, W).

        Returns
        -------
        torch.Tensor
            Scalar Focal loss.
        """
        log_p = F.log_softmax(logits, dim=1)  # (B, C, H, W)
        p = log_p.exp()                       # (B, C, H, W)

        # Clamp ignored targets to a valid class only for gather().
        t_clamped = targets.clone()
        t_clamped[targets == self.ignore_index] = 0

        log_pt = log_p.gather(1, t_clamped.unsqueeze(1)).squeeze(1)  # (B, H, W)
        pt = p.gather(1, t_clamped.unsqueeze(1)).squeeze(1)          # (B, H, W)

        focal_weight = (1.0 - pt) ** self.gamma                      # (B, H, W)
        wt = self.class_weights[t_clamped]                           # (B, H, W)

        loss = -focal_weight * wt * log_pt                           # (B, H, W)

        valid = (targets != self.ignore_index).float()
        loss = (loss * valid).sum() / valid.sum().clamp(min=1)

        return loss


# ---------------------------------------------------------------------------
# WIL – Wetland Imbalance Loss [Eq. 18, segformer_sae]
# lambda_dice = 0.5, lambda_focal = 0.5
# ---------------------------------------------------------------------------

class WIL(nn.Module):
    """
    Wetland Imbalance Loss, also usable as a generic imbalance loss.

    Formula
    -------
        L_WIL = lambda_dice × L_Dice + lambda_focal × L_Focal

    From the paper, section 3.5:

      - equal weighting is used;
      - lambda_Dice = 0.5;
      - lambda_Focal = 0.5;
      - gamma is set to 2.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        class_weights: Optional[torch.Tensor] = None,
        lambda_dice: float = 0.5,
        lambda_focal: float = 0.5,
        gamma: float = 2.0,
    ):
        """
        Initialize WIL.

        Parameters
        ----------
        num_classes :
            Number of semantic classes.
        ignore_index :
            Label value ignored during loss computation.
        class_weights :
            Optional tensor with shape (C,) used by FocalLoss.
        lambda_dice :
            Weight of the Dice component.
        lambda_focal :
            Weight of the Focal component.
        gamma :
            Focusing parameter used by FocalLoss.
        """
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal

        self.dice = DiceLoss(num_classes, ignore_index=ignore_index)
        self.focal = FocalLoss(
            num_classes,
            ignore_index=ignore_index,
            gamma=gamma,
            class_weights=class_weights,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Compute WIL and return a loss dictionary.

        Parameters
        ----------
        logits :
            Model logits with shape (B, C, H, W).
        targets :
            Ground-truth labels with shape (B, H, W).

        Returns
        -------
        dict
            Dictionary containing total loss and detached loss components.
        """
        l_dice = self.dice(logits, targets)
        l_focal = self.focal(logits, targets)
        total = self.lambda_dice * l_dice + self.lambda_focal * l_focal

        return {
            "loss": total,
            "dice": l_dice.detach(),
            "focal": l_focal.detach(),
        }


# ---------------------------------------------------------------------------
# Standard CrossEntropy for dofa, crossearth and DeepLabV3+
# ---------------------------------------------------------------------------

class CrossEntropyLoss(nn.Module):
    """
    Standard Cross-Entropy loss with ignore_index and optional class weights.

    Used by:

      - dofa       (mmseg config: CrossEntropyLoss, loss_weight=1.0)
      - crossearth (mmseg config: CrossEntropyLoss, loss_weight=1.0)
      - DeepLabV3+ (original Chen et al. 2018 setup)

    Parameters
    ----------
    ignore_index :
        Label value ignored during loss computation.
    class_weights :
        Optional tensor with shape (C,) for imbalanced classes.
        A common choice is 1 / freq_c, normalized.
    """

    def __init__(
        self,
        ignore_index: int = 255,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Initialize the Cross-Entropy loss.

        Parameters
        ----------
        ignore_index :
            Label value ignored during loss computation.
        class_weights :
            Optional tensor with shape (C,) containing class weights.
        """
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Compute Cross-Entropy loss and return a loss dictionary.

        Parameters
        ----------
        logits :
            Model logits with shape (B, C, H, W).
        targets :
            Ground-truth labels with shape (B, H, W).

        Returns
        -------
        dict
            Dictionary containing total loss and detached CE component.
        """
        loss = self.ce(logits, targets)
        return {
            "loss": loss,
            "ce": loss.detach(),
        }


# ---------------------------------------------------------------------------
# Factory: build_loss
# ---------------------------------------------------------------------------

def build_loss(
    model_type: str,
    num_classes: int = 14,
    ignore_index: int = 255,
    class_weights: Optional[torch.Tensor] = None,
    # WIL specific
    lambda_dice: float = 0.5,
    lambda_focal: float = 0.5,
    gamma: float = 2.0,
) -> nn.Module:
    """
    Build the correct loss function for the selected model.

    Mapping
    -------
        segformer_sae -> WIL  (Dice + Focal, lambda=0.5/0.5, gamma=2)
        deeplabv3plus -> CrossEntropy with optional class_weights
        dofa          -> CrossEntropy
        crossearth    -> CrossEntropy

    Parameters
    ----------
    model_type :
        Model type/name.
    num_classes :
        Number of semantic classes.
    ignore_index :
        Label value ignored during loss computation.
    class_weights :
        Optional tensor with shape (C,) containing class weights.
    lambda_dice :
        Dice component weight, used only by WIL.
    lambda_focal :
        Focal component weight, used only by WIL.
    gamma :
        Focal gamma, used only by WIL.

    Returns
    -------
    nn.Module
        Loss module returning a dictionary with at least the key "loss".

    Raises
    ------
    ValueError
        If model_type is not recognized.
    """
    mt = model_type.lower()

    if mt == "segformer_sae":
        return WIL(
            num_classes=num_classes,
            ignore_index=ignore_index,
            class_weights=class_weights,
            lambda_dice=lambda_dice,
            lambda_focal=lambda_focal,
            gamma=gamma,
        )

    if mt in ("deeplabv3plus", "dofa", "crossearth"):
        return CrossEntropyLoss(
            ignore_index=ignore_index,
            class_weights=class_weights,
        )

    raise ValueError(
        f"model_type '{model_type}' not recognized. "
        f"Choose one of: segformer_sae, deeplabv3plus, dofa, crossearth"
    )