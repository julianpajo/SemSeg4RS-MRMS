"""
train/train.py
--------------

Unified training script for all models.

Supported models:
  - SegFormerSAE
  - DeepLabV3+
  - DOFA
  - CrossEarth

Dataset
-------
This script uses preprocessing/dataset.py, therefore preprocessing is model-aware:

    model_name="segformer_sae"  -> RGBNIR, 512x512
    model_name="deeplabv3plus"  -> RGBNIR, 512x512
    model_name="dofa"           -> all_bands, 224x224
    model_name="crossearth"     -> RGBNIR, 504x504 if configured in preprocess.py

Training labels
---------------
    raw 0 -> 255 ignore_index
    raw 1 -> 0   sealed_soil
    raw 2 -> 1   non_sealed_soil

Therefore:

    num_classes = 2
    loss = CrossEntropyLoss(ignore_index=255)

Usage
-----
    python train/train.py --config configs/training/crossearth_spot.yaml

Resume
------
    python train/train.py --config configs/training/crossearth_spot.yaml --resume outputs/checkpoints/crossearth/last.pth

Device
------
    python train/train.py --config configs/training/crossearth_spot.yaml --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Project root path
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------

from losses import build_loss

from utils import (
    build_optimizer,
    build_scheduler,
    SegMetrics,
    AverageMeter,
    save_checkpoint,
    load_checkpoint,
)

from preprocessing.dataset import MultiSensorSegDataset
from preprocessing.collate import (
    SensorBatchSampler,
    segmentation_collate,
    dofa_pad_collate,
    ModelAdapter,
)
from preprocessing.transforms import (
    SegmentationTrainTransform,
    SegmentationEvalTransform,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML file.

    Parameters
    ----------
    path :
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed YAML content.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_samples(path_or_list):
    """
    Load a list of samples.

    Accepted inputs:

      - YAML path;
      - already loaded list.

    Expected sample format:

        - image: data/raw/spot/images/tile_001.tif
          label: data/raw/spot/labels/tile_001.tif
          sensor: spot
          sensor_config: configs/sensors/spot.yaml

    Parameters
    ----------
    path_or_list :
        YAML path or already loaded sample list.

    Returns
    -------
    list
        List of sample dictionaries.
    """
    if isinstance(path_or_list, str):
        return load_yaml(path_or_list)

    return path_or_list


def get_model_type(cfg: Dict[str, Any]) -> str:
    """
    Return the model type from the configuration.

    Parameters
    ----------
    cfg :
        Training configuration dictionary.

    Returns
    -------
    str
        Lowercase model type.
    """
    return cfg["model"]["type"].lower()


def get_num_classes(cfg: Dict[str, Any]) -> int:
    """
    Return the number of semantic classes from the configuration.

    Parameters
    ----------
    cfg :
        Training configuration dictionary.

    Returns
    -------
    int
        Number of classes.
    """
    return int(cfg["model"].get("num_classes", 2))


def apply_train_mode_overrides(
    cfg: Dict[str, Any],
    train_mode: str,
) -> Dict[str, Any]:
    """
    Apply global overrides to select the training mode.

    Available modes:

      config :
          Use the YAML exactly as provided.

      finetune :
          Use pretrained weights where possible and freeze the backbone when
          expected by the model strategy.

      scratch :
          Disable pretrained weights and make the backbone trainable.

    CrossEarth behavior
    -------------------
    finetune:
        - DINOv2 pretrained;
        - frozen backbone;
        - trainable patch_embed;
        - trainable Rein;
        - trainable decoder.

    scratch:
        - DINOv2 architecture with random initialization;
        - trainable backbone;
        - trainable patch_embed;
        - trainable Rein;
        - trainable decoder.

    Parameters
    ----------
    cfg :
        Training configuration dictionary.
    train_mode :
        One of "config", "finetune", or "scratch".

    Returns
    -------
    dict
        Configuration with training-mode overrides applied.
    """
    train_mode = train_mode.lower()

    if train_mode not in {"config", "finetune", "scratch"}:
        raise ValueError(
            f"Invalid train_mode: {train_mode}. "
            "Use: config | finetune | scratch"
        )

    cfg = dict(cfg)
    cfg["resolved_train_mode"] = train_mode

    if train_mode == "config":
        return cfg

    model_cfg = cfg["model"]
    model_type = model_cfg["type"].lower()

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------
    if train_mode == "finetune":
        if model_type == "crossearth":
            model_cfg["pretrained_backbone"] = True
            model_cfg["freeze_backbone"] = True
            model_cfg["train_patch_embed"] = True
            model_cfg["patch_embed_init"] = model_cfg.get(
                "patch_embed_init",
                "rgb_mean",
            )

        elif model_type == "dofa":
            model_cfg["pretrained"] = True
            model_cfg["freeze_backbone"] = True

        elif model_type == "segformer_sae":
            model_cfg["pretrained_backbone"] = True

        elif model_type == "deeplabv3plus":
            # Warning: if in_channels=4, pretrained_backbone=True only works
            # if DeepLab adapts the first convolution layer.
            model_cfg["pretrained_backbone"] = bool(
                model_cfg.get("pretrained_backbone", False)
            )

    # ------------------------------------------------------------------
    # Training from scratch
    # ------------------------------------------------------------------
    elif train_mode == "scratch":
        if model_type == "crossearth":
            model_cfg["pretrained_backbone"] = False
            model_cfg["freeze_backbone"] = False
            model_cfg["train_patch_embed"] = True
            model_cfg["patch_embed_init"] = "random"

        elif model_type == "dofa":
            model_cfg["pretrained"] = False
            model_cfg["freeze_backbone"] = False

        elif model_type == "segformer_sae":
            model_cfg["pretrained_backbone"] = False

        elif model_type == "deeplabv3plus":
            model_cfg["pretrained_backbone"] = False

    return cfg


def make_output_dir(path: str | Path) -> None:
    """
    Create an output directory if it does not already exist.

    Parameters
    ----------
    path :
        Directory path to create.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """
    Instantiate the model from the YAML configuration.

    Parameters
    ----------
    cfg :
        Training configuration dictionary.

    Returns
    -------
    nn.Module
        Initialized model.

    Raises
    ------
    ValueError
        If the model type is unknown.
    """
    model_type = get_model_type(cfg)
    model_cfg = cfg["model"]

    num_classes = int(model_cfg.get("num_classes", 2))

    # ------------------------------------------------------------------
    # SegFormer SAE
    # ------------------------------------------------------------------
    if model_type == "segformer_sae":
        from models.segformer_sae import SegFormerSAE

        if model_cfg.get("pretrained_backbone", False):
            model = SegFormerSAE.from_pretrained(
                hf_model_name=model_cfg.get("hf_name", "nvidia/mit-b2"),
                in_channels=int(model_cfg.get("in_channels", 4)),
                num_classes=num_classes,
                use_brd=bool(model_cfg.get("use_brd", True)),
            )
        else:
            model = SegFormerSAE(
                variant=model_cfg.get("variant", "mit-b2"),
                in_channels=int(model_cfg.get("in_channels", 4)),
                num_classes=num_classes,
                use_brd=bool(model_cfg.get("use_brd", True)),
            )

    # ------------------------------------------------------------------
    # DeepLabV3+
    # ------------------------------------------------------------------
    elif model_type == "deeplabv3plus":
        from models.deeplabv3plus import DeepLabV3Plus

        in_channels = int(model_cfg.get("in_channels", 4))

        model = DeepLabV3Plus(
            backbone=model_cfg.get("backbone", "resnet101"),
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained_backbone=bool(
                model_cfg.get("pretrained_backbone", in_channels == 3)
            ),
        )

    # ------------------------------------------------------------------
    # DOFA
    # ------------------------------------------------------------------
    elif model_type == "dofa":
        from models.dofa import DOFASeg

        model = DOFASeg(
            variant=model_cfg.get("variant", "base"),
            num_classes=num_classes,
            pretrained=bool(model_cfg.get("pretrained", True)),
            freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
            decoder=model_cfg.get("decoder", "mla"),
        )

    # ------------------------------------------------------------------
    # CrossEarth
    # ------------------------------------------------------------------
    elif model_type == "crossearth":
        from models.crossearth import CrossEarthSeg

        # In this project CrossEarth is RGBNIR by default.
        # Therefore, in_channels should be 4 unless an RGB baseline is explicitly required.
        in_channels = int(model_cfg.get("in_channels", 4))

        use_pretrained = bool(model_cfg.get("pretrained_backbone", True))
        backbone_ckpt = model_cfg.get("backbone_ckpt", None)

        if use_pretrained and backbone_ckpt is None:
            model = CrossEarthSeg.from_pretrained(
                variant=model_cfg.get("variant", "dinov2_vitl14_reg"),
                num_classes=num_classes,
                decoder=model_cfg.get("decoder", "mla"),
                decoder_dim=model_cfg.get("decoder_dim", None),
                num_tokens=int(model_cfg.get("num_tokens", 100)),
                token_dim=int(model_cfg.get("token_dim", 256)),
                dropout=float(model_cfg.get("dropout", 0.1)),
                in_channels=in_channels,
                patch_embed_init=model_cfg.get("patch_embed_init", "rgb_mean"),
                freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
                train_patch_embed=bool(model_cfg.get("train_patch_embed", True)),
                force_reload=bool(model_cfg.get("force_reload", False)),
            )

        else:
            model = CrossEarthSeg(
                variant=model_cfg.get("variant", "dinov2_vitl14_reg"),
                num_classes=num_classes,
                decoder=model_cfg.get("decoder", "mla"),
                decoder_dim=model_cfg.get("decoder_dim", None),
                num_tokens=int(model_cfg.get("num_tokens", 100)),
                token_dim=int(model_cfg.get("token_dim", 256)),
                dropout=float(model_cfg.get("dropout", 0.1)),
                in_channels=in_channels,
                patch_embed_init=model_cfg.get("patch_embed_init", "rgb_mean"),
                freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
                train_patch_embed=bool(model_cfg.get("train_patch_embed", True)),
            )

            if backbone_ckpt:
                model.load_backbone_checkpoint(backbone_ckpt)
            else:
                # If pretrained weights are disabled, load only the DINOv2 architecture
                # with pretrained=False.
                model.backbone_rein.load_backbone(pretrained=False)

    else:
        raise ValueError(f"Unknown model: '{model_type}'")

    return model


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def build_optimizer_for_model(
    model: nn.Module,
    train_cfg: Dict[str, Any],
) -> torch.optim.Optimizer:
    """
    Build the optimizer.

    If the model exposes parameter_groups(), those groups are used.

    This is important for CrossEarth because it allows different learning rates
    for:

      - patch_embed;
      - Rein;
      - decoder;
      - optional backbone_extra.

    Otherwise, the standard build_optimizer() from train.utils is used.

    Parameters
    ----------
    model :
        Model to optimize.
    train_cfg :
        Training configuration section.

    Returns
    -------
    torch.optim.Optimizer
        Initialized optimizer.

    Raises
    ------
    ValueError
        If the optimizer name is unsupported.
    """
    opt_name = train_cfg.get("optimizer", "adamw").lower()

    if hasattr(model, "parameter_groups"):
        groups = model.parameter_groups(
            lr_patch_embed=float(train_cfg.get("lr_patch_embed", 1e-5)),
            lr_rein=float(train_cfg.get("lr_rein", 1e-4)),
            lr_decoder=float(train_cfg.get("lr_decoder", 1e-3)),
            lr_backbone=float(train_cfg.get("lr_backbone", 1e-5)),
            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        )

        if opt_name == "adamw":
            return torch.optim.AdamW(groups)

        if opt_name == "adam":
            return torch.optim.Adam(groups)

        if opt_name == "sgd":
            return torch.optim.SGD(
                groups,
                momentum=float(train_cfg.get("momentum", 0.9)),
            )

        raise ValueError(f"Unsupported optimizer: {opt_name}")

    return build_optimizer(model, train_cfg)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_transforms(cfg: Dict[str, Any], split: str):
    """
    Build train/validation transforms.

    Augmentation can be disabled with:

        data:
          use_augmentation: false

    Parameters
    ----------
    cfg :
        Training configuration dictionary.
    split :
        Dataset split, either "train" or "val".

    Returns
    -------
    callable
        Transform callable.
    """
    data_cfg = cfg.get("data", {})

    if split == "train":
        if not bool(data_cfg.get("use_augmentation", True)):
            return SegmentationEvalTransform()

        aug_cfg = data_cfg.get("augmentation", {})

        return SegmentationTrainTransform(
            p_hflip=float(aug_cfg.get("p_hflip", 0.5)),
            p_vflip=float(aug_cfg.get("p_vflip", 0.5)),
            p_rotate=float(aug_cfg.get("p_rotate", 0.5)),
            brightness=float(aug_cfg.get("brightness", 0.05)),
            contrast=float(aug_cfg.get("contrast", 0.10)),
            noise_std=float(aug_cfg.get("noise_std", 0.01)),
        )

    return SegmentationEvalTransform()


def build_datasets(cfg: Dict[str, Any]):
    """
    Build train and validation datasets.

    Supported modes
    ---------------
    1. data.preprocessed: false

       Uses MultiSensorSegDataset and reads raw GeoTIFF files.
       Preprocessing is performed online.

    2. data.preprocessed: true

       Uses ProcessedSegDataset and reads already preprocessed GeoTIFF files.
       Preprocessing is NOT repeated.

    Parameters
    ----------
    cfg :
        Training configuration dictionary.

    Returns
    -------
    tuple
        train_ds, val_ds.
    """
    data_cfg = cfg["data"]
    model_type = get_model_type(cfg)

    train_samples = load_samples(data_cfg["train_samples"])
    val_samples = load_samples(data_cfg["val_samples"])

    use_preprocessed = bool(data_cfg.get("preprocessed", False))

    print(f"[train] Data mode: {'offline/preprocessed' if use_preprocessed else 'online/raw'}")

    # ------------------------------------------------------------------
    # Offline / preprocessed mode
    # ------------------------------------------------------------------
    if use_preprocessed:
        from preprocessing.processed_dataset import ProcessedSegDataset

        train_ds = ProcessedSegDataset(
            samples=train_samples,
            transform=build_transforms(cfg, split="train"),
            return_meta=bool(data_cfg.get("return_meta", True)),
        )

        val_ds = ProcessedSegDataset(
            samples=val_samples,
            transform=build_transforms(cfg, split="val"),
            return_meta=bool(data_cfg.get("return_meta", True)),
        )

        return train_ds, val_ds

    # ------------------------------------------------------------------
    # Online / raw GeoTIFF mode
    # ------------------------------------------------------------------

    model_name = data_cfg.get("model_name", model_type)

    patch_size_px = data_cfg.get("patch_size_px", None)
    patch_size_m = data_cfg.get("patch_size_m", None)

    if patch_size_px is None and patch_size_m is None:
        patch_size_px = 512

    train_ds = MultiSensorSegDataset(
        samples=train_samples,
        model_name=model_name,
        split="train",
        patch_size_px=patch_size_px,
        patch_size_m=patch_size_m,
        stride_px=None,
        max_invalid_frac=float(data_cfg.get("max_invalid_frac", 0.9)),
        min_valid_frac=float(data_cfg.get("min_valid_frac", 0.1)),
        min_valid_classes=int(data_cfg.get("min_valid_classes", 1)),
        max_retries=int(data_cfg.get("max_retries", 20)),
        remap_labels=bool(data_cfg.get("remap_labels", True)),
        ignore_index=int(cfg.get("loss", {}).get("ignore_index", 255)),
        transform=build_transforms(cfg, split="train"),
        return_meta=bool(data_cfg.get("return_meta", True)),
    )

    val_ds = MultiSensorSegDataset(
        samples=val_samples,
        model_name=model_name,
        split="val",
        patch_size_px=patch_size_px,
        patch_size_m=patch_size_m,
        stride_px=data_cfg.get("stride_px", patch_size_px),
        max_invalid_frac=float(data_cfg.get("max_invalid_frac", 0.9)),
        min_valid_frac=float(data_cfg.get("min_valid_frac", 0.1)),
        min_valid_classes=int(data_cfg.get("min_valid_classes", 1)),
        max_retries=1,
        remap_labels=bool(data_cfg.get("remap_labels", True)),
        ignore_index=int(cfg.get("loss", {}).get("ignore_index", 255)),
        transform=build_transforms(cfg, split="val"),
        return_meta=bool(data_cfg.get("return_meta", True)),
    )

    return train_ds, val_ds


def build_loaders(cfg: Dict[str, Any], train_ds, val_ds):
    """
    Build DataLoaders.

    For standard models:

        segmentation_collate

    For DOFA with sensors having different numbers of bands:

        dofa_pad_collate if data.use_dofa_padding = true

    Parameters
    ----------
    cfg :
        Training configuration dictionary.
    train_ds :
        Training dataset.
    val_ds :
        Validation dataset.

    Returns
    -------
    tuple
        train_loader, val_loader.
    """
    model_type = get_model_type(cfg)
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    batch_size = int(train_cfg.get("batch_size", 4))
    num_workers = int(train_cfg.get("num_workers", 4))
    pin_memory = bool(train_cfg.get("pin_memory", torch.cuda.is_available()))
    drop_last = bool(train_cfg.get("drop_last", True))

    use_sensor_batch_sampler = bool(data_cfg.get("sensor_batch_sampler", True))

    if model_type == "dofa" and bool(data_cfg.get("use_dofa_padding", False)):
        collate_fn = dofa_pad_collate
        use_sensor_batch_sampler = False
    else:
        collate_fn = segmentation_collate

    if use_sensor_batch_sampler:
        train_sampler = SensorBatchSampler(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
        )

        val_sampler = SensorBatchSampler(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            val_ds,
            batch_sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Train / Val loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    adapter: ModelAdapter,
    device: torch.device,
    epoch: int,
    log_every: int = 20,
    grad_clip: float = 1.0,
    amp_enabled: bool = False,
) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Parameters
    ----------
    model :
        Model to train.
    loader :
        Training DataLoader.
    criterion :
        Loss module returning a dictionary with at least key "loss".
    optimizer :
        Optimizer.
    scheduler :
        Learning-rate scheduler, stepped after each optimizer step.
    adapter :
        ModelAdapter used to move batches to device and call the model.
    device :
        Training device.
    epoch :
        Current epoch index.
    log_every :
        Logging frequency in steps.
    grad_clip :
        Maximum gradient norm. If <= 0, gradient clipping is disabled.
    amp_enabled :
        If True, use automatic mixed precision.

    Returns
    -------
    dict
        Average training losses for the epoch.
    """
    model.train()

    meters: Dict[str, AverageMeter] = {}

    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    for step, batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits, labels = adapter.forward(model, batch)
            loss_dict = criterion(logits, labels)
            loss = loss_dict["loss"]

        scaler.scale(loss).backward()

        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        batch_size = labels.shape[0]

        for key, value in loss_dict.items():
            if key not in meters:
                meters[key] = AverageMeter()

            meters[key].update(float(value.detach().cpu()), batch_size)

        if (step + 1) % log_every == 0:
            if scheduler is not None:
                lr = scheduler.get_last_lr()[0]
            else:
                lr = optimizer.param_groups[0]["lr"]

            log = "  ".join(
                f"{key}={meter.avg:.4f}"
                for key, meter in meters.items()
            )

            print(
                f"  Epoch {epoch:03d} | "
                f"step {step + 1:04d}/{len(loader)} | "
                f"{log}  lr={lr:.2e}"
            )

    return {
        key: meter.avg
        for key, meter in meters.items()
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    adapter: ModelAdapter,
    device: torch.device,
    num_classes: int,
    ignore_index: int = 255,
    amp_enabled: bool = False,
) -> Dict[str, Any]:
    """
    Validate the model.

    Parameters
    ----------
    model :
        Model to evaluate.
    loader :
        Validation DataLoader.
    criterion :
        Loss module returning a dictionary with at least key "loss".
    adapter :
        ModelAdapter used to move batches to device and call the model.
    device :
        Evaluation device.
    num_classes :
        Number of semantic classes.
    ignore_index :
        Label value ignored by metrics and loss.
    amp_enabled :
        If True, use automatic mixed precision.

    Returns
    -------
    dict
        Validation losses and segmentation metrics.
    """
    model.eval()

    metrics = SegMetrics(
        num_classes=num_classes,
        ignore_index=ignore_index,
    )

    meters: Dict[str, AverageMeter] = {}

    for batch in loader:
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits, labels = adapter.forward(model, batch)
            loss_dict = criterion(logits, labels)

        batch_size = labels.shape[0]

        for key, value in loss_dict.items():
            if key not in meters:
                meters[key] = AverageMeter()

            meters[key].update(float(value.detach().cpu()), batch_size)

        preds = logits.argmax(dim=1)
        metrics.update(preds, labels)

    results = metrics.compute()

    for key, meter in meters.items():
        results[key] = meter.avg

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    CLI entry point.

    The function loads the YAML config, applies the selected training mode,
    builds datasets, dataloaders, model, loss, optimizer and scheduler, then
    runs the full training/validation loop with checkpointing.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config.",
    )

    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint to resume from.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Device, e.g. cuda:0 or cpu.",
    )

    parser.add_argument(
        "--train-mode",
        default="config",
        choices=["config", "finetune", "scratch"],
        help=(
            "Training mode. "
            "'config' uses the YAML; "
            "'finetune' uses pretrained/frozen settings where expected; "
            "'scratch' disables pretrained weights and makes the backbone trainable."
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    cfg = load_yaml(args.config)
    cfg = apply_train_mode_overrides(cfg, args.train_mode)

    model_type = get_model_type(cfg)
    num_classes = get_num_classes(cfg)

    print(f"[train] Train mode: {cfg.get('resolved_train_mode', 'config')}")

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    if args.device is not None:
        device = torch.device(args.device)
    elif cfg.get("device") is not None:
        device = torch.device(cfg["device"])
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    print(f"[train] Device: {device}")

    # ------------------------------------------------------------------
    # Dataset / Loader
    # ------------------------------------------------------------------

    print("\n[train] Building datasets...")

    train_ds, val_ds = build_datasets(cfg)

    print("[train] Train dataset:")
    print(train_ds.describe())

    print("[train] Val dataset:")
    print(val_ds.describe())

    train_loader, val_loader = build_loaders(
        cfg=cfg,
        train_ds=train_ds,
        val_ds=val_ds,
    )

    print(f"[train] Train loader steps: {len(train_loader)}")
    print(f"[train] Val loader steps  : {len(val_loader)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    print("\n[train] Building model...")

    model = build_model(cfg).to(device)

    print(f"[train] Model type: {model_type}")

    if hasattr(model, "count_parameters"):
        print("[train] Parameters:")
        for key, value in model.count_parameters().items():
            print(f"  {key:<30}: {value:>12,}")

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    loss_cfg = cfg.get("loss", {})
    ignore_index = int(loss_cfg.get("ignore_index", 255))

    criterion = build_loss(
        model_type=model_type,
        num_classes=num_classes,
        ignore_index=ignore_index,
        lambda_dice=float(loss_cfg.get("lambda_dice", 0.5)),
        lambda_focal=float(loss_cfg.get("lambda_focal", 0.5)),
        gamma=float(loss_cfg.get("gamma", 2.0)),
        class_weights=loss_cfg.get("class_weights", None),
    ).to(device)

    print(f"[train] Loss: {criterion.__class__.__name__}")

    # ------------------------------------------------------------------
    # Optimizer / Scheduler
    # ------------------------------------------------------------------

    train_cfg = cfg["training"]

    epochs = int(train_cfg["epochs"])
    steps_total = epochs * len(train_loader)

    if steps_total <= 0:
        raise RuntimeError(
            "steps_total <= 0. Check dataset, batch_size and sampler."
        )

    warmup_steps = int(
        train_cfg.get(
            "warmup_steps",
            min(500, max(1, steps_total // 10)),
        )
    )

    optimizer = build_optimizer_for_model(model, train_cfg)
    scheduler = build_scheduler(
        optimizer,
        steps_total,
        warmup_steps=warmup_steps,
    )

    print(f"[train] Optimizer: {optimizer.__class__.__name__}")
    print(f"[train] Scheduler warmup steps: {warmup_steps}")
    print(f"[train] Total steps: {steps_total}")

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    start_epoch = 0
    best_miou = 0.0

    output_dir = Path(train_cfg.get("output_dir", "outputs"))
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", output_dir / "checkpoints"))

    train_mode = cfg.get("resolved_train_mode", "config")

    run_name = train_cfg.get(
        "run_name",
        f"{model_type}_{train_mode}",
    )

    ckpt_base = ckpt_dir / run_name

    make_output_dir(ckpt_base)

    if args.resume is not None:
        ckpt = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            str(device),
        )

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_miou = float(ckpt.get("metrics", {}).get("miou", 0.0))

        print(f"[train] Resumed from epoch {start_epoch}, best_miou={best_miou:.4f}")

    # ------------------------------------------------------------------
    # Adapter
    # ------------------------------------------------------------------

    adapter = ModelAdapter(
        model_type=model_type,
        device=device,
        use_band_mask=bool(cfg.get("data", {}).get("use_dofa_padding", False)),
    )

    # ------------------------------------------------------------------
    # AMP
    # ------------------------------------------------------------------

    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"

    print(f"[train] AMP enabled: {amp_enabled}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    print(
        f"\n[train] Start training: "
        f"{epochs} epochs, {len(train_loader)} steps/epoch\n"
    )

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            adapter=adapter,
            device=device,
            epoch=epoch,
            log_every=int(train_cfg.get("log_every", 20)),
            grad_clip=float(train_cfg.get("grad_clip", 1.0)),
            amp_enabled=amp_enabled,
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            adapter=adapter,
            device=device,
            num_classes=num_classes,
            ignore_index=ignore_index,
            amp_enabled=amp_enabled,
        )

        elapsed = time.time() - t0

        train_log = "  ".join(
            f"train_{key}={value:.4f}"
            for key, value in train_metrics.items()
        )

        val_log = "  ".join(
            f"val_{key}={value:.4f}"
            for key, value in val_metrics.items()
            if key != "per_class"
        )

        print(
            f"\nEpoch {epoch:03d}/{epochs - 1:03d} | "
            f"{train_log} | {val_log} | {elapsed:.0f}s\n"
        )

        if (epoch + 1) % int(train_cfg.get("print_per_class_every", 5)) == 0:
            if "per_class" in val_metrics:
                iou_str = "  ".join(
                    f"c{i}={value:.3f}"
                    for i, value in enumerate(val_metrics["per_class"])
                )
                print(f"  per-class IoU: {iou_str}")

        # ------------------------------------------------------------------
        # Checkpoint
        # ------------------------------------------------------------------

        current_miou = float(val_metrics.get("miou", 0.0))
        is_best = current_miou > best_miou

        if is_best:
            best_miou = current_miou

        metrics_to_save = {
            key: value
            for key, value in val_metrics.items()
            if key != "per_class"
        }

        save_checkpoint(
            path=str(ckpt_base / "last.pth"),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=metrics_to_save,
            cfg=cfg,
        )

        if is_best:
            save_checkpoint(
                path=str(ckpt_base / "best.pth"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics_to_save,
                cfg=cfg,
            )

            print(f"  ★ New best mIoU: {best_miou:.4f}")

    print(f"\n[train] Training completed. Best mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()