"""
pl_modules/components/base.py
------------------------------
Abstract base classes for training components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
import torch.nn as nn


class Loss(ABC):
    """Base class for loss functions."""

    @abstractmethod
    def build(self) -> nn.Module:
        ...


class Optimizer(ABC):
    """Base class for optimizers."""

    @abstractmethod
    def build(self) -> Callable[[nn.Module], torch.optim.Optimizer]:
        """Returns a callable: model -> optimizer."""
        ...


class Scheduler(ABC):
    """Base class for lr schedulers."""

    @abstractmethod
    def build(self) -> Callable[[torch.optim.Optimizer], Any]:
        """Returns a callable: optimizer -> scheduler."""
        ...

    @property
    def interval(self) -> str:
        """'step' or 'epoch'. Override in subclasses that need step-level scheduling."""
        return "epoch"
