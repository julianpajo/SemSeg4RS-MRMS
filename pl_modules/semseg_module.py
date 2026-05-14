# pl_modules/semseg_module.py

from __future__ import annotations

from typing import Any, Dict

import lightning.pytorch as pl
import torch

from models.factory import build_model, build_optimizer_for_model, get_model_type, get_num_classes
from models.losses import build_loss
from models.utils import build_scheduler, SegMetrics
from preprocessing.collate import ModelAdapter


class SemSegLightningModule(pl.LightningModule):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.model_type = get_model_type(cfg)
        self.num_classes = get_num_classes(cfg)

        loss_cfg = cfg.get("loss", {})
        self.ignore_index = int(loss_cfg.get("ignore_index", 255))

        self.model = build_model(cfg)

        self.criterion = build_loss(
            model_type=self.model_type,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            lambda_dice=float(loss_cfg.get("lambda_dice", 0.5)),
            lambda_focal=float(loss_cfg.get("lambda_focal", 0.5)),
            gamma=float(loss_cfg.get("gamma", 2.0)),
            class_weights=loss_cfg.get("class_weights", None),
        )

        self.adapter = ModelAdapter(
            model_type=self.model_type,
            device=torch.device("cpu"),
            use_band_mask=bool(cfg.get("data", {}).get("use_dofa_padding", False)),
        )

        self.val_metrics = SegMetrics(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

    def forward(self, batch):
        self.adapter.device = self.device
        logits, labels = self.adapter.forward(self.model, batch)
        return logits, labels

    def training_step(self, batch, batch_idx):
        logits, labels = self.forward(batch)
        loss_dict = self.criterion(logits, labels)

        self.log("train/loss", loss_dict["loss"], prog_bar=True, on_step=True, on_epoch=True)

        for key, value in loss_dict.items():
            if key != "loss":
                self.log(f"train/{key}", value, on_step=False, on_epoch=True)

        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        logits, labels = self.forward(batch)
        loss_dict = self.criterion(logits, labels)

        preds = logits.argmax(dim=1)
        self.val_metrics.update(preds, labels)

        self.log("val/loss", loss_dict["loss"], prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()

        self.log("val/miou", metrics["miou"], prog_bar=True)
        self.log("val/pixel_acc", metrics["pixel_acc"], prog_bar=True)

        for i, value in enumerate(metrics["per_class"]):
            self.log(f"val/iou_class_{i}", value)

        self.val_metrics.reset()

    def configure_optimizers(self):
        train_cfg = self.cfg["training"]

        optimizer = build_optimizer_for_model(
            model=self.model,
            train_cfg=train_cfg,
        )

        steps_total = int(self.trainer.estimated_stepping_batches)
        warmup_steps = int(train_cfg.get("warmup_steps", 100))

        scheduler = build_scheduler(
            optimizer=optimizer,
            num_steps=max(1, steps_total),
            warmup_steps=warmup_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }