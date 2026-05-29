"""
tasks/segmentation_lightning.py
--------------------------------
High-level training task wrapper for Lightning AI Studios.

This module provides ``LightningAISegTask``, a convenience class that
assembles a fully configured Lightning ``Trainer`` with:

- local checkpoint saving;
- optional Lightning AI model registry upload;
- early stopping;
- learning-rate monitoring.

Use ``upload_checkpoints_to_registry=False`` to keep experiment charts/logs
while avoiding heavy checkpoint uploads to Lightning Weights.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from litmodels.integrations import LightningModelCheckpoint


class LightningAISegTask:
    """
    Training task for semantic segmentation on Lightning AI Studios.

    The task can operate in two checkpoint modes:

    - local-only checkpointing with ``ModelCheckpoint``;
    - registry checkpointing with ``LightningModelCheckpoint``.

    Experiment logging is controlled by the passed logger. Therefore, using a
    Lightning logger with ``log_model=False`` still keeps charts/logs while
    avoiding model artifact upload.
    """

    def __init__(
        self,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        output_dir: str,
        organization: str,
        teamspace: str,
        model_registry_name: str,
        logger: Any,
        max_epochs: int = 50,
        n_gradient_accumul: int = 1,
        es_patience: int = 10,
        monitor_metric: str = "val_miou",
        monitor_mode: str = "max",
        checkpoint_topk: int = 1,
        upload_checkpoints_to_registry: bool = False,
        save_weights_only: bool = True,
        devices: str | int | list[int] = "auto",
        accelerator: str = "auto",
        log_every_n_steps: int = 1,
        precision: str = "bf16-mixed",
    ) -> None:
        self.model = model
        self.datamodule = datamodule
        self.output_dir = str(output_dir)
        self.organization = str(organization)
        self.teamspace = str(teamspace)
        self.registry_name = str(model_registry_name)
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.upload_checkpoints_to_registry = upload_checkpoints_to_registry

        os.makedirs(self.output_dir, exist_ok=True)
        torch.backends.cudnn.benchmark = False

        checkpoint_dir = Path(self.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        filename = f"best-epoch={{epoch:02d}}-{monitor_metric}={{{monitor_metric}:.4f}}"

        if upload_checkpoints_to_registry:
            registry_path = (
                f"{self.organization}/{self.teamspace}/{self.registry_name}"
            )

            self.checkpoint_callback = LightningModelCheckpoint(
                model_registry=registry_path,
                monitor=monitor_metric,
                mode=monitor_mode,
                save_top_k=checkpoint_topk,
                save_last=False,
                save_weights_only=save_weights_only,
                filename=filename,
            )
        else:
            self.checkpoint_callback = ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                monitor=monitor_metric,
                mode=monitor_mode,
                save_top_k=checkpoint_topk,
                save_last=False,
                save_weights_only=save_weights_only,
                filename=filename,
                auto_insert_metric_name=False,
            )

        early_stopping = EarlyStopping(
            monitor=monitor_metric,
            mode=monitor_mode,
            patience=es_patience,
            min_delta=0.0,
        )

        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        self.trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            logger=logger,
            callbacks=[
                self.checkpoint_callback,
                early_stopping,
                lr_monitor,
            ],
            enable_checkpointing=True,
            log_every_n_steps=log_every_n_steps,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0,
            accumulate_grad_batches=n_gradient_accumul,
            default_root_dir=self.output_dir,
            precision="bf16-mixed",
        )

        print(
            f"  Dashboard   : "
            f"https://lightning.ai/{self.organization}/{self.teamspace}/experiments"
        )

        if self.upload_checkpoints_to_registry:
            print(
                f"  Registry    : "
                f"https://lightning.ai/{self.organization}/{self.teamspace}/models/"
                f"{self.registry_name}"
            )
        else:
            print(f"  Registry    : disabled")
            print(f"  Checkpoints : {checkpoint_dir}")

    def train(self) -> None:
        """
        Run the training loop and print a timing and results summary.
        """
        t0 = time.perf_counter()

        self.trainer.fit(
            model=self.model,
            datamodule=self.datamodule,
        )

        elapsed = time.perf_counter() - t0

        print(f"\n{'=' * 60}")
        print(f"Training complete in {elapsed:.1f}s")
        print(f"Best {self.monitor_metric} : {self.checkpoint_callback.best_model_score}")
        print(f"Best ckpt     : {self.checkpoint_callback.best_model_path}")

        print(
            f"Experiment    : "
            f"https://lightning.ai/{self.organization}/{self.teamspace}/experiments"
        )

        if self.upload_checkpoints_to_registry:
            print(
                f"Registry      : "
                f"https://lightning.ai/{self.organization}/{self.teamspace}/models/"
                f"{self.registry_name}"
            )
        else:
            print("Registry      : disabled")
            print(f"Local output  : {self.output_dir}")

        print("=" * 60)

    def run(self) -> None:
        """
        Alias for ``train()``.
        """
        self.train()

    @property
    def best_model_path(self) -> str:
        """
        Path of the best checkpoint saved during training.
        """
        return self.checkpoint_callback.best_model_path

    @property
    def best_model_score(self):
        """
        Value of the monitored metric at the best checkpoint.
        """
        return self.checkpoint_callback.best_model_score