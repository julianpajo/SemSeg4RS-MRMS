"""
pl_modules/modules/dofa/dofa_trainer.py
----------------------------------------
Training entry point for DOFA on a single sensor dataset.

DOFA (Dynamic One-For-All) is a wave-dynamic remote sensing foundation
model that accepts images with any number of spectral bands alongside
their central wavelengths in micrometers.

All experiment hyperparameters are loaded from:

    configs/training_configs/dofa.py
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from litlogger import LightningLogger

from tasks.segmentation_lightning import LightningAISegTask

from pl_modules.components import AdamW, CrossEntropyLoss, CosineAnnealing
from pl_modules.modules.dofa.dofa_lightning import (
    DOFALightning,
    DOFALightningConfig,
)
from datamodules.semseg_datamodule import SemSegStreamingDataModule

from configs.training_configs.dofa import get_config


# ============================================================
# Config
# ============================================================

CFG = get_config(dataset_name="spot")

torch.set_float32_matmul_precision(CFG.runtime.matmul_precision)


# ============================================================
# Builders
# ============================================================

def build_datamodule() -> SemSegStreamingDataModule:
    """
    Instantiate the streaming datamodule for training and validation.

    Reads pre-optimized LitData chunks produced by ``datasets/opt_dataset.py``.

    DOFA requires wavelengths, so no RGBNIR-only collate function is used here.
    The full-band image tensor and wavelengths are kept in the batch.
    """
    return SemSegStreamingDataModule(
        train_dir=str(CFG.paths.train_chunks),
        val_dir=str(CFG.paths.val_chunks),
        batch_size=CFG.data.batch_size,
        num_workers=CFG.data.num_workers,
        shuffle_train=True,
        drop_last=True,
    )


def build_lightning_model() -> DOFALightning:
    """
    Build the DOFA Lightning model with optimizer and scheduler.

    Unlike CrossEarth, DeepLabV3+ and SegFormer-SAE, ``max_iters`` is not
    needed here because CosineAnnealing is stepped per epoch.
    """
    criterion = CrossEntropyLoss(
        ignore_index=CFG.task.ignore_index,
    ).build()

    optimizer_fn = AdamW(
        lr=CFG.optim.lr,
        weight_decay=CFG.optim.weight_decay,
    ).build()

    scheduler_fn = CosineAnnealing(
        epochs=CFG.runtime.epochs,
        min_lr=0.0,
    ).build()

    config = DOFALightningConfig(
        # Task
        num_classes=CFG.task.num_classes,
        ignore_index=CFG.task.ignore_index,

        # Model
        variant=CFG.model.variant,
        pretrained=CFG.model.use_pretrained,
        freeze_backbone=CFG.model.freeze_backbone,
        decoder=CFG.model.decoder,

        # Training components
        criterion=criterion,
        optimizer_fn=optimizer_fn,
        scheduler_fn=scheduler_fn,
        scheduler_interval=CFG.optim.scheduler_interval,
    )

    return DOFALightning(config=config)


def build_logger() -> LightningLogger:
    """
    Build the Lightning AI experiment logger.

    Logs and charts are kept, but model artifact upload is disabled.
    Checkpoints are handled locally by ``LightningAISegTask``.
    """
    metadata: Dict[str, str] = {
        "model": str(CFG.model_name),
        "dataset": str(CFG.dataset_name),
        "variant": str(CFG.model.variant),
        "decoder": str(CFG.model.decoder),
        "num_classes": str(CFG.task.num_classes),
    }

    hparams: Dict[str, Any] = {
        "model": CFG.model_name,
        "dataset": CFG.dataset_name,
        "experiment_name": CFG.experiment_name,

        "variant": CFG.model.variant,
        "decoder": CFG.model.decoder,
        "num_classes": CFG.task.num_classes,
        "ignore_index": CFG.task.ignore_index,

        "batch_size": CFG.data.batch_size,
        "gradient_accumulation": CFG.data.gradient_accumulation,
        "effective_batch_size": (
            CFG.data.batch_size * CFG.data.gradient_accumulation
        ),
        "num_workers": CFG.data.num_workers,

        "epochs": CFG.runtime.epochs,
        "patience": CFG.runtime.patience,
        "log_every_n_steps": CFG.runtime.log_every_n_steps,

        "lr": CFG.optim.lr,
        "weight_decay": CFG.optim.weight_decay,
        "scheduler": "CosineAnnealing",
        "scheduler_interval": CFG.optim.scheduler_interval,

        "loss": "CrossEntropyLoss",
        "use_pretrained": CFG.model.use_pretrained,
        "freeze_backbone": CFG.model.freeze_backbone,
    }

    logger = LightningLogger(
        root_dir=str(CFG.paths.output_dir),
        name=str(CFG.experiment_name),
        log_model=False,
        save_logs=True,
        metadata=metadata,
    )

    logger.log_hyperparams(hparams)

    return logger


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Entry point for DOFA training.
    """
    CFG.paths.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("DOFA TRAINING CONFIG")
    print("=" * 80)
    print(f"Dataset        : {CFG.dataset_name}")
    print(f"Model          : {CFG.model_name}")
    print(f"Experiment     : {CFG.experiment_name}")
    print(f"Train chunks   : {CFG.paths.train_chunks}")
    print(f"Val chunks     : {CFG.paths.val_chunks}")
    print(f"Output dir     : {CFG.paths.output_dir}")
    print(f"Registry       : disabled for checkpoints")
    print(f"Teamspace      : {CFG.registry.organization}/{CFG.registry.teamspace}")
    print("=" * 80)

    datamodule = build_datamodule()

    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()

    print(f"Train batches  : {len(train_loader)}")
    print(f"Epochs         : {CFG.runtime.epochs}")

    lit_model = build_lightning_model()
    logger = build_logger()

    task = LightningAISegTask(
        model=lit_model,
        datamodule=datamodule,
        output_dir=str(CFG.paths.output_dir),
        organization=str(CFG.registry.organization),
        teamspace=str(CFG.registry.teamspace),
        model_registry_name=str(CFG.registry.model_registry_name),
        logger=logger,
        max_epochs=CFG.runtime.epochs,
        log_every_n_steps=CFG.runtime.log_every_n_steps,
        n_gradient_accumul=CFG.data.gradient_accumulation,
        es_patience=CFG.runtime.patience,
        upload_checkpoints_to_registry=False,
    )

    task.run()

    print(f"Best checkpoint: {task.best_model_path}")
    print(f"Best val_miou:   {task.best_model_score}")


if __name__ == "__main__":
    main()