"""
train/run_train.py
--------------

Unified training script for all models.

Supported models:
  - SegFormerSAE
  - DeepLabV3+
  - DOFA
  - CrossEarth

Dataset
-------
This script uses datasets/raw.py, therefore datasets is model-aware:

    model_name="segformer_sae"  -> RGBNIR, 512x512
    model_name="deeplabv3plus"  -> RGBNIR, 512x512
    model_name="dofa"           -> all_bands, 224x224
    model_name="crossearth"     -> RGBNIR, 504x504 if configured in pipeline.py

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
    python train/run_train.py --config configs/training/crossearth_spot.yaml

Resume
------
    python train/run_train.py --config configs/training/crossearth_spot.yaml --resume outputs/checkpoints/crossearth/last.pth

Device
------
    python train/run_train.py --config configs/training/crossearth_spot.yaml --device cuda:0
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import yaml

from datasets.core.raw import MultiSensorSegDataset
from datasets.collate import (
    SensorBatchSampler,
    segmentation_collate,
    dofa_pad_collate,
)
from datasets.transforms.augment import (
    SegmentationTrainTransform,
    SegmentationEvalTransform,
)

# ---------------------------------------------------------------------------
# Project root path
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
    Build optimizer.

    If the model exposes parameter_groups(), only the arguments accepted by
    that specific model are passed.

    This avoids passing CrossEarth-specific arguments such as lr_patch_embed
    to models like DOFA.
    """
    import inspect

    opt_name = train_cfg.get("optimizer", "adamw").lower()
    weight_decay = float(train_cfg.get("weight_decay", 0.01))

    if hasattr(model, "parameter_groups"):
        sig = inspect.signature(model.parameter_groups)
        accepted = set(sig.parameters.keys())

        candidate_kwargs = {
            # Generic
            "lr": float(train_cfg.get("lr", 1e-4)),
            "lr_encoder": float(train_cfg.get("lr_backbone", train_cfg.get("lr", 1e-4) / 10)),
            "lr_backbone": float(train_cfg.get("lr_backbone", train_cfg.get("lr", 1e-4) / 10)),
            "lr_decoder": float(train_cfg.get("lr_decoder", train_cfg.get("lr", 1e-4))),
            "lr_head": float(train_cfg.get("lr", 1e-4)),
            "lr_sae": float(train_cfg.get("lr", 1e-4)),

            # CrossEarth-specific
            "lr_patch_embed": float(train_cfg.get("lr_patch_embed", 1e-5)),
            "lr_rein": float(train_cfg.get("lr_rein", 1e-4)),

            # Common
            "weight_decay": weight_decay,
        }

        kwargs = {
            k: v for k, v in candidate_kwargs.items()
            if k in accepted
        }

        groups = model.parameter_groups(**kwargs)

    else:
        lr = float(train_cfg.get("lr", 1e-4))
        groups = [
            {
                "params": model.parameters(),
                "lr": lr,
                "weight_decay": weight_decay,
            }
        ]

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
        Training datasets.
    val_ds :
        Validation datasets.

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
        from datasets.core.cached import ProcessedSegDataset

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