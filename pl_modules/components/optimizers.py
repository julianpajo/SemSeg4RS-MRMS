"""
pl_modules/components/optimizers.py
-------------------------------------
Optimizer components for semantic segmentation training.

Each optimizer is implemented as a configuration class that inherits from
``Optimizer`` and exposes a ``build()`` method returning a callable
``model -> torch.optim.Optimizer``. This pattern allows the optimizer to
be configured independently from model instantiation and integrates
cleanly with ``LightningSegConfig``.

Available components
--------------------
AdamW
    AdamW optimizer with automatic per-group learning rate support.
    If the model exposes a ``parameter_groups(**kwargs)`` method, per-group
    learning rates are resolved automatically by inspecting the method
    signature and passing only the accepted keyword arguments.

Usage
-----
    from pl_modules.components.optimizers import AdamW

    # All parameters share the same lr
    optimizer_fn = AdamW(lr=6e-5, weight_decay=0.05).build()
    optimizer = optimizer_fn(model)

    # Per-group lr (model must expose parameter_groups())
    optimizer_fn = AdamW(
        lr=6e-5,
        weight_decay=0.05,
        lr_backbone=6e-6,
        lr_patch_embed=6e-6,
        lr_rein=6e-5,
        lr_decoder=6e-4,
    ).build()
    optimizer = optimizer_fn(model)
"""

from __future__ import annotations

import inspect
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from .base import Optimizer


import inspect
from typing import Any, Callable, Dict, Optional, cast

import torch
import torch.nn as nn
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    AdamW optimizer builder.

    If the model exposes a callable `parameter_groups(**kwargs)` method,
    per-group learning rates are used automatically. Otherwise, all
    parameters share `lr`.
    """

    def __init__(
        self,
        lr: float = 6e-5,
        weight_decay: float = 0.05,
        lr_backbone: Optional[float] = None,
        lr_encoder: Optional[float] = None,
        lr_decoder: Optional[float] = None,
        lr_patch_embed: Optional[float] = None,
        lr_rein: Optional[float] = None,
        lr_sae: Optional[float] = None,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:

        """
        Parameters
        ----------
        lr : float
            Global learning rate, used as default for all parameter groups
            and as the base lr for AdamW when no parameter groups are defined.
        weight_decay : float
            L2 regularization coefficient.
        lr_backbone : float | None
            Learning rate for backbone parameters. If None, defaults to
            ``lr * 0.1``.
        lr_encoder : float | None
            Learning rate for encoder parameters. If None, defaults to
            ``lr * 0.1``.
        lr_decoder : float | None
            Learning rate for decoder parameters. If None, defaults to
            ``lr * 10``.
        lr_patch_embed : float | None
            Learning rate for patch embedding parameters. If None, defaults
            to ``lr * 0.1``.
        lr_rein : float | None
            Learning rate for Rein adapter parameters (CrossEarth). If None,
            defaults to ``lr``.
        lr_sae : float | None
            Learning rate for SAE module parameters (SegFormer-SAE). If None,
            defaults to ``lr * 10``.
        betas : tuple[float, float]
            Adam beta coefficients.
        eps : float
            Adam epsilon for numerical stability.
        """

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_backbone = lr_backbone
        self.lr_encoder = lr_encoder
        self.lr_decoder = lr_decoder
        self.lr_patch_embed = lr_patch_embed
        self.lr_rein = lr_rein
        self.lr_sae = lr_sae
        self.betas = betas
        self.eps = eps

    def build(self) -> Callable[[nn.Module], torch.optim.Optimizer]:
        """
        Build the optimizer factory.

        Returns a callable that accepts a ``torch.nn.Module`` and returns a
        configured ``torch.optim.AdamW`` instance.

        If the model exposes a ``parameter_groups(**kwargs)`` method, its
        signature is inspected and only the accepted keyword arguments are
        forwarded. This allows models such as ``CrossEarthSeg``, ``DOFASeg``,
        and ``SegFormerSAE`` to define custom per-group learning rates without
        any changes to this class.

        Returns
        -------
        Callable[[nn.Module], torch.optim.Optimizer]
            A function ``model -> optimizer``.
        """
        cfg = self

        def _build(model: nn.Module) -> torch.optim.Optimizer:
            parameter_groups = getattr(model, "parameter_groups", None)

            if callable(parameter_groups):
                parameter_groups_fn = cast(Callable[..., Any], parameter_groups)

                sig = inspect.signature(parameter_groups_fn)
                accepted = set(sig.parameters.keys())

                candidates: Dict[str, float] = {
                    "lr": cfg.lr,
                    "weight_decay": cfg.weight_decay,
                    "lr_backbone": (
                        cfg.lr_backbone
                        if cfg.lr_backbone is not None
                        else cfg.lr * 0.1
                    ),
                    "lr_encoder": (
                        cfg.lr_encoder
                        if cfg.lr_encoder is not None
                        else cfg.lr * 0.1
                    ),
                    "lr_decoder": (
                        cfg.lr_decoder
                        if cfg.lr_decoder is not None
                        else cfg.lr * 10.0
                    ),
                    "lr_patch_embed": (
                        cfg.lr_patch_embed
                        if cfg.lr_patch_embed is not None
                        else cfg.lr * 0.1
                    ),
                    "lr_rein": (
                        cfg.lr_rein
                        if cfg.lr_rein is not None
                        else cfg.lr
                    ),
                    "lr_sae": (
                        cfg.lr_sae
                        if cfg.lr_sae is not None
                        else cfg.lr * 10.0
                    ),
                }

                kwargs = {
                    key: value
                    for key, value in candidates.items()
                    if key in accepted
                }

                params = parameter_groups_fn(**kwargs)
            else:
                params = model.parameters()

            return torch.optim.AdamW(
                params,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                betas=cfg.betas,
                eps=cfg.eps,
            )

        return _build
