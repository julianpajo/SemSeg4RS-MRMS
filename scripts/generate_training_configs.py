"""
scripts/generate_training_configs.py
------------------------------------

Automatically generate training YAML configs for:

    models:
      - crossearth
      - deeplabv3plus
      - segformer_sae
      - dofa

    datasets / sensor setups:
      - planetscope
      - spot
      - mixed

    modes:
      - online
      - offline

Output
------
    configs/training/{model}/{dataset}/online.yaml
    configs/training/{model}/{dataset}/offline.yaml

Usage
-----
    python scripts/generate_training_configs.py

Overwrite existing files:

    python scripts/generate_training_configs.py --overwrite

Generate only one model:

    python scripts/generate_training_configs.py --models crossearth --overwrite

Generate only one dataset/setup:

    python scripts/generate_training_configs.py --datasets mixed --overwrite

Notes
-----
online.yaml:
    - uses raw split YAML files:
        configs/splits/raw/{dataset}/train_samples.yaml
        configs/splits/raw/{dataset}/val_samples.yaml

    - preprocessing is performed online by MultiSensorSegDataset.

offline.yaml:
    - uses processed split YAML files:
        configs/splits/processed/{model}/{dataset}/train_samples.yaml
        configs/splits/processed/{model}/{dataset}/val_samples.yaml

    - includes a `preprocessing` section used by:

        scripts/build_processed_dataset.py --config ... --split train|val|test

    - training uses ProcessedSegDataset.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

import yaml


# ---------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------

MODELS = [
    "crossearth",
    "deeplabv3plus",
    "segformer_sae",
    "dofa",
]

DATASETS = [
    "planetscope",
    "spot",
    "mixed",
]

SENSOR_CONFIG_ROOT = Path("configs/sensors")

DATASET_TO_SENSORS = {
    "planetscope": ["planetscope"],
    "spot": ["spot"],
    "mixed": ["planetscope", "spot"],
}


# ---------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------

def model_config(model_name: str) -> Dict[str, Any]:
    """
    Return the model section for a given model.

    Parameters
    ----------
    model_name :
        Name of the model.

    Returns
    -------
    dict
        YAML-ready model configuration.

    Raises
    ------
    ValueError
        If the model is not supported.
    """
    if model_name == "crossearth":
        return {
            "type": "crossearth",
            "variant": "dinov2_vitl14_reg",
            "num_classes": 2,
            "decoder": "mla",
            "decoder_dim": None,
            "in_channels": 4,
            "pretrained_backbone": True,
            "patch_embed_init": "rgb_mean",
            "freeze_backbone": True,
            "train_patch_embed": True,
            "num_tokens": 100,
            "token_dim": 256,
            "dropout": 0.1,
        }

    if model_name == "deeplabv3plus":
        return {
            "type": "deeplabv3plus",
            "backbone": "resnet101",
            "in_channels": 4,
            "num_classes": 2,
            "pretrained_backbone": False,
        }

    if model_name == "segformer_sae":
        return {
            "type": "segformer_sae",
            "variant": "mit-b2",
            "in_channels": 4,
            "num_classes": 2,
            "use_brd": True,
            "pretrained_backbone": False,
        }

    if model_name == "dofa":
        return {
            "type": "dofa",
            "variant": "base",
            "num_classes": 2,
            "pretrained": True,
            "freeze_backbone": True,
            "decoder": "mla",
        }

    raise ValueError(f"Unsupported model: {model_name}")


# ---------------------------------------------------------------------
# Data configs
# ---------------------------------------------------------------------

def online_data_config(model_name: str, dataset_name: str) -> Dict[str, Any]:
    """
    Build the data section for online training from raw GeoTIFF files.

    In online mode:

      - raw GeoTIFF files are read during training;
      - crop, band selection, normalization, resize and label remapping are
        performed online by MultiSensorSegDataset.

    Parameters
    ----------
    model_name :
        Target model name.
    dataset_name :
        Dataset/sensor setup name.

    Returns
    -------
    dict
        YAML-ready data configuration.
    """
    cfg = {
        "preprocessed": False,
        "model_name": model_name,
        "train_samples": f"configs/splits/raw/{dataset_name}/train_samples.yaml",
        "val_samples": f"configs/splits/raw/{dataset_name}/val_samples.yaml",
        "patch_size_px": 512,
        "stride_px": 512,
        "sensor_batch_sampler": True,
        "remap_labels": True,
        "max_invalid_frac": 0.9,
        "min_valid_frac": 0.1,
        "min_valid_classes": 1,
        "max_retries": 20,
        "use_augmentation": True,
        "augmentation": {
            "p_hflip": 0.5,
            "p_vflip": 0.5,
            "p_rotate": 0.5,
            "brightness": 0.05,
            "contrast": 0.10,
            "noise_std": 0.01,
        },
        "return_meta": True,
    }

    if model_name == "dofa" and dataset_name == "mixed":
        # If DOFA receives sensors with different numbers of bands,
        # either keep sensor_batch_sampler=True or enable DOFA padding if supported.
        cfg["use_dofa_padding"] = False

    return cfg


def offline_data_config(model_name: str, dataset_name: str) -> Dict[str, Any]:
    """
    Build the data section for offline training from preprocessed GeoTIFF files.

    In offline mode:

      - training reads preprocessed GeoTIFF files;
      - crop, band selection, normalization, resize and label remapping have
        already been performed by build_processed_dataset.py.

    Parameters
    ----------
    model_name :
        Target model name.
    dataset_name :
        Dataset/sensor setup name.

    Returns
    -------
    dict
        YAML-ready data configuration.
    """
    cfg = {
        "preprocessed": True,
        "model_name": model_name,
        "train_samples": (
            f"configs/splits/processed/{model_name}/{dataset_name}/train_samples.yaml"
        ),
        "val_samples": (
            f"configs/splits/processed/{model_name}/{dataset_name}/val_samples.yaml"
        ),
        "sensor_batch_sampler": True,
        "use_augmentation": True,
        "augmentation": {
            "p_hflip": 0.5,
            "p_vflip": 0.5,
            "p_rotate": 0.5,
            "brightness": 0.05,
            "contrast": 0.10,
            "noise_std": 0.01,
        },
        "return_meta": True,
    }

    if model_name == "dofa" and dataset_name == "mixed":
        cfg["use_dofa_padding"] = False

    return cfg


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

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_sensor_tile_size(sensor_name: str) -> tuple[int, int]:
    """
    Read tile_size_px from configs/sensors/{sensor}.yaml.

    Expected YAML field
    -------------------
        tile_size_px: [H, W]

    If the field is missing, the fallback is 512 x 512.

    Parameters
    ----------
    sensor_name :
        Sensor name.

    Returns
    -------
    tuple of int
        Tile size as (height, width).

    Raises
    ------
    ValueError
        If tile_size_px has an invalid format.
    """
    path = SENSOR_CONFIG_ROOT / f"{sensor_name}.yaml"

    if not path.exists():
        print(f"[WARNING] Missing sensor config: {path}. Using tile_size_px=[512, 512].")
        return 512, 512

    cfg = load_yaml(path)

    tile_size = cfg.get("tile_size_px", [512, 512])

    if isinstance(tile_size, int):
        return int(tile_size), int(tile_size)

    if isinstance(tile_size, list) and len(tile_size) == 2:
        return int(tile_size[0]), int(tile_size[1])

    raise ValueError(
        f"Invalid tile_size_px in {path}. Expected int or [H, W], got: {tile_size}"
    )


def estimate_patches_per_image(
    tile_h: int,
    tile_w: int,
    patch_size_px: int,
    max_patches_per_image: int,
) -> int:
    """
    Estimate the number of random training patches per raw tile.

    Examples
    --------
        tile 512 x 512,   patch 512 -> 1
        tile 1024 x 1024, patch 512 -> 4
        tile 2048 x 2048, patch 512 -> 16

    Parameters
    ----------
    tile_h :
        Tile height in pixels.
    tile_w :
        Tile width in pixels.
    patch_size_px :
        Training crop size in pixels.
    max_patches_per_image :
        Maximum allowed number of patches per image.

    Returns
    -------
    int
        Estimated number of patches per image.

    Raises
    ------
    ValueError
        If patch_size_px is not strictly positive.
    """
    if patch_size_px <= 0:
        raise ValueError(f"patch_size_px must be > 0, got {patch_size_px}")

    n_h = max(1, math.ceil(tile_h / patch_size_px))
    n_w = max(1, math.ceil(tile_w / patch_size_px))

    n = n_h * n_w

    return max(1, min(int(n), int(max_patches_per_image)))


def build_patches_per_sensor(
    dataset_name: str,
    patch_size_px: int,
    max_patches_per_image: int,
) -> Dict[str, int]:
    """
    Build a per-sensor patches_per_image dictionary.

    For mixed datasets, examples are:

        {
          "planetscope": 1,
          "spot": 1
        }

    or, with larger raw scenes:

        {
          "planetscope": 16,
          "spot": 4
        }

    Parameters
    ----------
    dataset_name :
        Dataset/sensor setup name.
    patch_size_px :
        Training crop size in pixels.
    max_patches_per_image :
        Maximum allowed number of patches per image.

    Returns
    -------
    dict
        Mapping from sensor name to patches_per_image.
    """
    sensors = DATASET_TO_SENSORS[dataset_name]

    out = {}

    for sensor_name in sensors:
        tile_h, tile_w = read_sensor_tile_size(sensor_name)

        out[sensor_name] = estimate_patches_per_image(
            tile_h=tile_h,
            tile_w=tile_w,
            patch_size_px=patch_size_px,
            max_patches_per_image=max_patches_per_image,
        )

    return out


# ---------------------------------------------------------------------
# Offline preprocessing configs
# ---------------------------------------------------------------------

def preprocessing_config(model_name: str, dataset_name: str) -> Dict[str, Any]:
    """
    Build the preprocessing section used by scripts/build_processed_dataset.py.

    This section is used only by offline configs.

    patches_per_image is computed automatically from:

      - configs/sensors/{sensor}.yaml -> tile_size_px
      - patch_size_px
      - max_patches_per_image

    For single-sensor datasets, patches_per_image is an integer.

    For mixed datasets, patches_per_image is set to "auto" and a per-sensor
    dictionary is also written.

    Parameters
    ----------
    model_name :
        Target model name.
    dataset_name :
        Dataset/sensor setup name.

    Returns
    -------
    dict
        YAML-ready preprocessing configuration.
    """
    patch_size_px = 512
    max_patches_per_image = 20

    patches_by_sensor = build_patches_per_sensor(
        dataset_name=dataset_name,
        patch_size_px=patch_size_px,
        max_patches_per_image=max_patches_per_image,
    )

    if dataset_name in {"planetscope", "spot"}:
        patches_per_image = patches_by_sensor[dataset_name]
    else:
        patches_per_image = "auto"

    return {
        "raw_train_samples": (
            f"configs/splits/raw/{dataset_name}/train_samples.yaml"
        ),
        "raw_val_samples": (
            f"configs/splits/raw/{dataset_name}/val_samples.yaml"
        ),
        "raw_test_samples": (
            f"configs/splits/raw/{dataset_name}/test_samples.yaml"
        ),

        "processed_train_samples": (
            f"configs/splits/processed/{model_name}/{dataset_name}/train_samples.yaml"
        ),
        "processed_val_samples": (
            f"configs/splits/processed/{model_name}/{dataset_name}/val_samples.yaml"
        ),
        "processed_test_samples": (
            f"configs/splits/processed/{model_name}/{dataset_name}/test_samples.yaml"
        ),

        "out_root": "data/processed",

        # Raw crop size before model-aware resize.
        "patch_size_px": patch_size_px,

        # Used for val/test grid extraction.
        "stride_px": patch_size_px,

        # Used only for train random patch extraction.
        # If dataset is mixed, build_processed_dataset.py will use
        # patches_per_image_by_sensor.
        "patches_per_image": patches_per_image,
        "patches_per_image_by_sensor": patches_by_sensor,
        "max_patches_per_image": max_patches_per_image,

        "max_retries": 50,
        "max_invalid_frac": 0.9,
        "min_valid_frac": 0.1,
        "min_valid_classes": 1,

        "seed": 42,
        "skip_existing": True,
        "stop_on_error": False,
    }


# ---------------------------------------------------------------------
# Training configs
# ---------------------------------------------------------------------

def training_config(
    model_name: str,
    dataset_name: str,
    mode: str,
) -> Dict[str, Any]:
    """
    Build default training hyperparameters.

    These are safe defaults. Tune batch_size, epochs and learning rates
    depending on GPU memory and dataset size.

    Parameters
    ----------
    model_name :
        Target model name.
    dataset_name :
        Dataset/sensor setup name.
    mode :
        Training mode, either "online" or "offline".

    Returns
    -------
    dict
        YAML-ready training configuration.
    """
    cfg = {
        "epochs": 30,
        "batch_size": 1,
        "num_workers": 0,
        "pin_memory": True,
        "drop_last": True,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "grad_clip": 1.0,
        "amp": True,
        "log_every": 1,
        "print_per_class_every": 5,
        "output_dir": "outputs",
        "checkpoint_dir": "outputs/checkpoints",
    }

    if model_name == "crossearth":
        cfg.update(
            {
                "lr_patch_embed": 1.0e-5,
                "lr_rein": 1.0e-4,
                "lr_decoder": 1.0e-3,
                "lr_backbone": 1.0e-5,
            }
        )

    elif model_name == "deeplabv3plus":
        cfg.update(
            {
                "lr": 1.0e-4,
                "batch_size": 2,
            }
        )

    elif model_name == "segformer_sae":
        cfg.update(
            {
                "lr": 1.0e-4,
                "batch_size": 2,
            }
        )

    elif model_name == "dofa":
        cfg.update(
            {
                "lr": 1.0e-4,
                "lr_backbone": 1.0e-5,
                "batch_size": 1,
            }
        )

    # Mixed datasets are usually heavier and safer with smaller batches.
    if dataset_name == "mixed":
        cfg["batch_size"] = min(int(cfg["batch_size"]), 1)

    return cfg


def loss_config(model_name: str) -> Dict[str, Any]:
    """
    Build the loss section.

    build_loss() decides the model-specific loss internally.
    For segformer_sae, it can use WIL if implemented.

    Parameters
    ----------
    model_name :
        Target model name.

    Returns
    -------
    dict
        YAML-ready loss configuration.
    """
    cfg = {
        "ignore_index": 255,
    }

    if model_name == "segformer_sae":
        cfg.update(
            {
                "lambda_dice": 0.5,
                "lambda_focal": 0.5,
                "gamma": 2.0,
            }
        )

    return cfg


# ---------------------------------------------------------------------
# Full config
# ---------------------------------------------------------------------

def build_config(
    model_name: str,
    dataset_name: str,
    mode: str,
) -> Dict[str, Any]:
    """
    Build a full YAML training config.

    Parameters
    ----------
    model_name :
        Target model name.
    dataset_name :
        Dataset/sensor setup name.
    mode :
        Training mode, either "online" or "offline".

    Returns
    -------
    dict
        Full YAML-ready configuration.

    Raises
    ------
    ValueError
        If mode is invalid.
    """
    if mode not in {"online", "offline"}:
        raise ValueError(f"Invalid mode: {mode}")

    cfg = {
        "device": "cuda",
        "model": model_config(model_name),
        "data": (
            online_data_config(model_name, dataset_name)
            if mode == "online"
            else offline_data_config(model_name, dataset_name)
        ),
        "loss": loss_config(model_name),
        "training": training_config(model_name, dataset_name, mode),
    }

    # Only offline configs need the preprocessing section.
    # Online configs perform preprocessing directly inside MultiSensorSegDataset.
    if mode == "offline":
        cfg["preprocessing"] = preprocessing_config(
            model_name=model_name,
            dataset_name=dataset_name,
        )

    return cfg


# ---------------------------------------------------------------------
# Write utilities
# ---------------------------------------------------------------------

def save_yaml(
    data: Dict[str, Any],
    path: Path,
    overwrite: bool = False,
) -> bool:
    """
    Save data to a YAML file.

    Parameters
    ----------
    data :
        Data structure to serialize.
    path :
        Output YAML path.
    overwrite :
        If False, existing files are not overwritten.

    Returns
    -------
    bool
        True if the file was written, False if it was skipped.
    """
    if path.exists() and not overwrite:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    return True


def generate_configs(
    out_root: Path,
    overwrite: bool,
    models: List[str],
    datasets: List[str],
) -> None:
    """
    Generate training configuration files.

    Parameters
    ----------
    out_root :
        Output root folder.
    overwrite :
        If True, overwrite existing YAML files.
    models :
        List of model names to generate.
    datasets :
        List of dataset/setup names to generate.
    """
    written = 0
    skipped = 0

    for model_name in models:
        for dataset_name in datasets:
            for mode in ["online", "offline"]:
                cfg = build_config(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    mode=mode,
                )

                out_path = (
                    out_root
                    / model_name
                    / dataset_name
                    / f"{mode}.yaml"
                )

                did_write = save_yaml(
                    data=cfg,
                    path=out_path,
                    overwrite=overwrite,
                )

                if did_write:
                    written += 1
                    print(f"[WRITE] {out_path}")
                else:
                    skipped += 1
                    print(f"[SKIP ] {out_path}")

    print("\n" + "=" * 80)
    print("CONFIG GENERATION DONE")
    print("=" * 80)
    print("Written:", written)
    print("Skipped:", skipped)
    print("Output root:", out_root)
    print("=" * 80)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out-root",
        type=str,
        default="configs/training",
        help="Output root for training configs.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing YAML files.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        choices=MODELS,
        help="Models to generate configs for.",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        choices=DATASETS,
        help="Datasets/setups to generate configs for.",
    )

    return parser.parse_args()


def main() -> None:
    """
    CLI entry point.

    Generate YAML configs for the selected model/dataset combinations.
    """
    args = parse_args()

    generate_configs(
        out_root=Path(args.out_root),
        overwrite=bool(args.overwrite),
        models=args.models,
        datasets=args.datasets,
    )


if __name__ == "__main__":
    main()