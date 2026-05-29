"""
pl_modules/modules/segformersae/segformersae_trainer.py
--------------------------------------------------------
Training entry point for SegFormer-SAE on a single sensor dataset.

SegFormer-SAE extends the standard SegFormer architecture with a
Spectral Awareness Embedding (SAE) module that projects multispectral
or RGBNIR input into the first encoder stage, and an optional
Boundary-aware Refinement Decoder (BRD).

All experiment hyperparameters are loaded from:

    configs/training_configs/segformer_sae.py
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from litlogger import LightningLogger

from tasks.segmentation_lightning import LightningAISegTask

from pl_modules.components import AdamW, ImbalanceLoss, PolyLR
from pl_modules.modules.segformersae.segformersae_lightning import (
    SegFormerSAELightning,
    SegFormerSAELightningConfig,
)
from datamodules.semseg_datamodule import SemSegStreamingDataModule

from configs.training_configs.segformersae import get_config


# ============================================================
# Config
# ============================================================

CFG = get_config(dataset_name="spot")

torch.set_float32_matmul_precision(CFG.runtime.matmul_precision)


# ============================================================
# Collate
# ============================================================

def rgbnir_collate(batch: List[Dict]) -> Dict:
    """
    Collate function for SegFormer-SAE.

    Pipeline:
        1. Select RGBNIR bands from the full-band image tensor using rgbnir_idx.
        2. Stack image and label tensors.
        3. Return only fields that should be forwarded downstream.

    Input item format:
        image       : FloatTensor (C, H, W)  — all bands
        label       : LongTensor  (H, W)
        rgbnir_idx  : LongTensor  (4,)       — R, G, B, NIR indices
        wavelengths : FloatTensor (C,)       — ignored by SegFormer-SAE

    Output:
        image : FloatTensor (B, 4, H, W)
        label : LongTensor  (B, H, W)
    """
    images = torch.stack(
        [
            item["image"][item["rgbnir_idx"]]  # (C, H, W) -> (4, H, W)
            for item in batch
        ]
    )

    labels = torch.stack(
        [
            item["label"]
            for item in batch
        ]
    )

    result: Dict = {
        "image": images,
        "label": labels,
    }

    # rgbnir_idx is consumed here.
    # wavelengths is only needed by DOFA and must not be forwarded.
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
    A collate function applies RGBNIR band selection at batch-assembly time.
    """
    return SemSegStreamingDataModule(
        train_dir=str(CFG.paths.train_chunks),
        val_dir=str(CFG.paths.val_chunks),
        batch_size=CFG.data.batch_size,
        num_workers=CFG.data.num_workers,
        shuffle_train=True,
        drop_last=True,
        collate_fn=rgbnir_collate,
    )


def build_lightning_model(max_iters: int) -> SegFormerSAELightning:
    """
    Build the SegFormer-SAE Lightning model with optimizer and scheduler.
    """
    in_channels = CFG.task.in_channels

    if in_channels is None:
        raise ValueError("SegFormer-SAE requires CFG.task.in_channels to be set.")

    if CFG.task.class_weights is None:
        raise ValueError("SegFormer-SAE requires CFG.task.class_weights to be set.")

    criterion = ImbalanceLoss(
        num_classes=CFG.task.num_classes,
        ignore_index=CFG.task.ignore_index,
        alpha=CFG.optim.loss_alpha,
        class_weights=CFG.task.class_weights,
    ).build()

    optimizer_fn = AdamW(
        lr=CFG.optim.lr,
        weight_decay=CFG.optim.weight_decay,
        lr_encoder=CFG.optim.lr_encoder,
        lr_decoder=CFG.optim.lr_decoder,
        lr_sae=CFG.optim.lr_sae,
    ).build()

    scheduler_fn = PolyLR(
        max_iters=max_iters,
        warmup_iters=CFG.optim.warmup_iters,
    ).build()

    config = SegFormerSAELightningConfig(
        # Task
        num_classes=CFG.task.num_classes,
        ignore_index=CFG.task.ignore_index,
        in_channels=in_channels,

        # Model
        variant=CFG.model.variant,
        hf_model_name=CFG.model.hf_model_name,
        use_pretrained_backbone=CFG.model.use_pretrained_backbone,
        use_brd=CFG.model.use_brd,

        # Training components
        criterion=criterion,
        optimizer_fn=optimizer_fn,
        scheduler_fn=scheduler_fn,
        scheduler_interval=CFG.optim.scheduler_interval,
    )

    return SegFormerSAELightning(config=config)


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
        "hf_model_name": str(CFG.model.hf_model_name),
        "in_channels": str(CFG.task.in_channels),
        "num_classes": str(CFG.task.num_classes),
        "use_brd": str(CFG.model.use_brd),
    }

    hparams: Dict[str, Any] = {
        "model": CFG.model_name,
        "dataset": CFG.dataset_name,
        "experiment_name": CFG.experiment_name,

        "variant": CFG.model.variant,
        "hf_model_name": CFG.model.hf_model_name,
        "num_classes": CFG.task.num_classes,
        "ignore_index": CFG.task.ignore_index,
        "class_weights": CFG.task.class_weights,
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
        "lr_encoder": CFG.optim.lr_encoder,
        "lr_decoder": CFG.optim.lr_decoder,
        "lr_sae": CFG.optim.lr_sae,
        "weight_decay": CFG.optim.weight_decay,
        "scheduler": "PolyLR",
        "scheduler_interval": CFG.optim.scheduler_interval,
        "warmup_iters": CFG.optim.warmup_iters,

        "loss": "ImbalanceLoss",
        "loss_alpha": CFG.optim.loss_alpha,
        "use_pretrained_backbone": CFG.model.use_pretrained_backbone,
        "use_brd": CFG.model.use_brd,
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
    Entry point for SegFormer-SAE training.
    """
    CFG.paths.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("SEGFORMER-SAE TRAINING CONFIG")
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

    max_iters = len(train_loader) * CFG.runtime.epochs

    print(f"Train batches  : {len(train_loader)}")
    print(f"Epochs         : {CFG.runtime.epochs}")
    print(f"Max iters      : {max_iters}")

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