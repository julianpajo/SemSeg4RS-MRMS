"""
pl_modules/modules/crossearth/crossearth_trainer.py
-----------------------------------------------------

Training entry point for CrossEarth on a single sensor dataset.

This script assembles the full training pipeline for CrossEarth with a
Mask2Former decoder, including streaming data loading, model configuration,
optimizer and scheduler setup, experiment logging, and checkpoint
registration to the Lightning AI model registry.

All experiment hyperparameters are loaded from:

    training_configs/crossearth.py
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torchvision.transforms.functional import center_crop

from litlogger import LightningLogger

from tasks.segmentation_lightning import LightningAISegTask

from pl_modules.components import AdamW, Mask2FormerLossConfig, PolyLR
from pl_modules.modules.crossearth.crossearth_lightning import (
    CrossEarthLightning,
    CrossEarthLightningConfig,
)
from datamodules.semseg_datamodule import SemSegStreamingDataModule

from configs.training_configs.crossearth import get_config


# ============================================================
# Config
# ============================================================

CFG = get_config(dataset_name="spot")

torch.set_float32_matmul_precision(CFG.runtime.matmul_precision)


# ============================================================
# Collate
# ============================================================

def crossearth_collate(batch: List[Dict]) -> Dict:
    """
    Collate function for CrossEarth.

    Pipeline:
        1. Select RGBNIR bands from the full-band image tensor using rgbnir_idx.
        2. Center-crop image and label from 512 to 504.
        3. Return only the fields required by CrossEarth.

    Input item format:
        image       : FloatTensor (C, H, W)  — all bands
        label       : LongTensor  (H, W)
        rgbnir_idx  : LongTensor  (4,)       — R, G, B, NIR indices
        wavelengths : FloatTensor (C,)       — ignored by CrossEarth

    Output:
        image : FloatTensor (B, 4, 504, 504)
        label : LongTensor  (B, 504, 504)
    """
    images = torch.stack(
        [
            center_crop(
                item["image"][item["rgbnir_idx"]],  # (C, H, W) -> (4, H, W)
                CFG.model.crossearth_patch_size,
            )
            for item in batch
        ]
    )

    labels = torch.stack(
        [
            center_crop(
                item["label"].unsqueeze(0),          # (H, W) -> (1, H, W)
                CFG.model.crossearth_patch_size,
            ).squeeze(0)                             # (1, H, W) -> (H, W)
            for item in batch
        ]
    )

    result: Dict = {
        "image": images,
        "label": labels,
    }

    # rgbnir_idx is consumed inside this collate function.
    # wavelengths is only needed by DOFA and must not be forwarded to CrossEarth.
    skip = {
        "image",
        "label",
        "rgbnir_idx",
        "wavelengths",
    }

    for key in batch[0]:
        if key in skip:
            continue

        values = [item[key] for item in batch]

        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = values

    return result


# ============================================================
# Builders
# ============================================================

def build_datamodule() -> SemSegStreamingDataModule:
    """
    Instantiate the streaming datamodule for training and validation.

    Reads pre-optimized LitData chunks produced by ``datasets/opt_dataset.py``.
    A collate function applies the CrossEarth-specific RGBNIR selection and
    center-crop 512 -> 504 at batch-assembly time.
    """
    return SemSegStreamingDataModule(
        train_dir=str(CFG.paths.train_chunks),
        val_dir=str(CFG.paths.val_chunks),
        batch_size=CFG.data.batch_size,
        num_workers=CFG.data.num_workers,
        shuffle_train=True,
        drop_last=True,
        collate_fn=crossearth_collate,
    )


def build_criterion() -> torch.nn.Module:
    """
    Build the Mask2Former loss for CrossEarth training.
    """
    return Mask2FormerLossConfig(
        num_classes=CFG.task.num_classes,
        ignore_index=CFG.task.ignore_index,
        loss_cls_weight=2.0,
        loss_mask_weight=5.0,
        loss_dice_weight=5.0,
        bg_cls_weight=0.1,
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        aux_loss_weight=1.0,
    ).build()


def build_lightning_model(max_iters: int) -> CrossEarthLightning:
    """
    Build the CrossEarth Lightning model with optimizer and scheduler.
    """
    criterion = build_criterion()

    optimizer_fn = AdamW(
        lr=CFG.optim.lr,
        weight_decay=CFG.optim.weight_decay,
        lr_backbone=CFG.optim.lr_backbone,
        lr_patch_embed=CFG.optim.lr_patch_embed,
        lr_rein=CFG.optim.lr_rein,
        lr_decoder=CFG.optim.lr_decoder,
    ).build()

    scheduler_fn = PolyLR(
        max_iters=max_iters,
        warmup_iters=CFG.optim.warmup_iters,
    ).build()

    in_channels = CFG.task.in_channels

    if in_channels is None:
        raise ValueError("CrossEarth requires CFG.task.in_channels to be set.")

    config = CrossEarthLightningConfig(
        # Task
        num_classes=CFG.task.num_classes,
        ignore_index=CFG.task.ignore_index,
        in_channels=in_channels,

        # Model
        variant=CFG.model.variant,
        decoder_dim=CFG.model.decoder_dim,
        num_tokens=CFG.model.num_tokens,
        token_dim=CFG.model.token_dim,
        dropout=CFG.model.dropout,
        patch_embed_init=CFG.model.patch_embed_init,

        # Backbone
        use_pretrained_backbone=CFG.model.use_pretrained_backbone,
        freeze_backbone=CFG.model.freeze_backbone,
        train_patch_embed=CFG.model.train_patch_embed,
        force_reload=CFG.model.force_reload,

        # Training components
        criterion=criterion,
        optimizer_fn=optimizer_fn,
        scheduler_fn=scheduler_fn,
        scheduler_interval=CFG.optim.scheduler_interval,
    )

    return CrossEarthLightning(config=config)


def build_logger() -> LightningLogger:
    """
    Build the Lightning AI experiment logger.
    """
    metadata: Dict[str, str] = {
        "model": str(CFG.model_name),
        "dataset": str(CFG.dataset_name),
        "variant": str(CFG.model.variant),
        "decoder": str(CFG.model.decoder),
        "in_channels": str(CFG.task.in_channels),
        "num_classes": str(CFG.task.num_classes),
    }

    logger = LightningLogger(
        root_dir=str(CFG.paths.output_dir),
        name=str(CFG.experiment_name),
        log_model=False,
        save_logs=True,
        metadata=metadata,
    )

    logger.log_hyperparams(
        {
            "model": CFG.model_name,
            "dataset": CFG.dataset_name,
            "experiment_name": CFG.experiment_name,

            "variant": CFG.model.variant,
            "decoder": CFG.model.decoder,
            "num_classes": CFG.task.num_classes,
            "ignore_index": CFG.task.ignore_index,
            "in_channels": CFG.task.in_channels,

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
            "lr_patch_embed": CFG.optim.lr_patch_embed,
            "lr_rein": CFG.optim.lr_rein,
            "lr_decoder": CFG.optim.lr_decoder,
            "lr_backbone": CFG.optim.lr_backbone,
            "weight_decay": CFG.optim.weight_decay,
            "scheduler": "PolyLR",
            "scheduler_interval": CFG.optim.scheduler_interval,
            "warmup_iters": CFG.optim.warmup_iters,

            "loss": "Mask2FormerLoss",

            "use_pretrained_backbone": CFG.model.use_pretrained_backbone,
            "freeze_backbone": CFG.model.freeze_backbone,
            "train_patch_embed": CFG.model.train_patch_embed,
            "patch_embed_init": CFG.model.patch_embed_init,
            "force_reload": CFG.model.force_reload,

            "num_tokens": CFG.model.num_tokens,
            "token_dim": CFG.model.token_dim,
            "dropout": CFG.model.dropout,
            "crossearth_patch_size": CFG.model.crossearth_patch_size,
        }
    )

    return logger


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Entry point for CrossEarth training.
    """
    CFG.paths.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("CROSSEARTH TRAINING CONFIG")
    print("=" * 80)
    print(f"Dataset        : {CFG.dataset_name}")
    print(f"Model          : {CFG.model_name}")
    print(f"Experiment     : {CFG.experiment_name}")
    print(f"Train chunks   : {CFG.paths.train_chunks}")
    print(f"Val chunks     : {CFG.paths.val_chunks}")
    print(f"Output dir     : {CFG.paths.output_dir}")
    print(f"Registry       : {CFG.registry.model_registry_name}")
    print(f"Teamspace      : {CFG.registry.organization}/{CFG.registry.teamspace}")
    print("=" * 80)

    datamodule = build_datamodule()

    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()

    max_iters = len(train_loader) * CFG.runtime.epochs

    lit_model = build_lightning_model(max_iters=max_iters)
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