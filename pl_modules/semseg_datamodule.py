# pl_modules/semseg_datamodule.py

from __future__ import annotations

from typing import Any, Dict

import lightning.pytorch as pl

from models.factory import build_datasets, build_loaders


class SemSegDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage=None):
        self.train_ds, self.val_ds = build_datasets(self.cfg)

    def train_dataloader(self):
        train_loader, _ = build_loaders(
            cfg=self.cfg,
            train_ds=self.train_ds,
            val_ds=self.val_ds,
        )
        return train_loader

    def val_dataloader(self):
        _, val_loader = build_loaders(
            cfg=self.cfg,
            train_ds=self.train_ds,
            val_ds=self.val_ds,
        )
        return val_loader