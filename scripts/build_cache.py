"""
scripts/build_cache.py
----------------------------------

Build an offline preprocessed GeoTIFF datasets from raw GeoTIFF splits.

Recommended usage
-----------------
Use the offline training YAML as the single source of truth:

    python scripts/build_cache.py \
        --config configs/training/crossearth/planetscope/offline.yaml \
        --split train

    python scripts/build_cache.py \
        --config configs/training/crossearth/planetscope/offline.yaml \
        --split val

    python scripts/build_cache.py \
        --config configs/training/crossearth/planetscope/offline.yaml \
        --split test

The script reads the `datasets` section from the YAML:

    datasets:
      raw_train_samples: configs/splits/raw/planetscope/train_samples.yaml
      raw_val_samples: configs/splits/raw/planetscope/val_samples.yaml
      raw_test_samples: configs/splits/raw/planetscope/test_samples.yaml

      processed_train_samples: configs/splits/processed/crossearth/planetscope/train_samples.yaml
      processed_val_samples: configs/splits/processed/crossearth/planetscope/val_samples.yaml
      processed_test_samples: configs/splits/processed/crossearth/planetscope/test_samples.yaml

      out_root: data/processed

      patch_size_px: 512
      stride_px: 512
      patches_per_image: 20

      max_retries: 50
      max_invalid_frac: 0.9
      min_valid_frac: 0.1
      min_valid_classes: 1

      seed: 42
      skip_existing: true

Legacy usage
------------
The old explicit CLI arguments are still supported:

    python scripts/build_cache.py \
        --samples configs/splits/raw/planetscope/train_samples.yaml \
        --model-name crossearth \
        --split train \
        --out-root data/processed \
        --out-samples configs/splits/processed/crossearth/planetscope/train_samples.yaml \
        --patch-size-px 512 \
        --patches-per-image 20 \
        --seed 42 \
        --skip-existing

Output image GeoTIFF
--------------------
    - multiband
    - float32
    - range approximately [0, 1]
    - example CrossEarth shape: 4 x 504 x 504

Output label GeoTIFF
--------------------
    - single-band
    - uint8
    - values:
        0   = sealed_soil
        1   = non_sealed_soil
        255 = ignore_index
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import rasterio
import yaml
from rasterio.transform import Affine
from tqdm import tqdm

from configs.sensor_configs import read_sensor
from datasets.core.raw import infer_sensor_config_path

from datasets.preprocessing.pipeline import (
    IGNORE_INDEX,
    preprocess_image_for_model,
    preprocess_label_for_model,
)

from datasets.sampling.patches import (
    raster_size,
    read_image_window,
    read_label_window,
    make_grid_items,
    sample_random_valid_windows,
)


# ---------------------------------------------------------------------
# YAML utilities
# ---------------------------------------------------------------------

def load_yaml(path: str | Path):
    """
    Load a YAML file.

    Parameters
    ----------
    path :
        Path to the YAML file.

    Returns
    -------
    Any
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


def save_yaml(data, path: str | Path) -> None:
    """
    Save data to a YAML file.

    Parameters
    ----------
    data :
        Data structure to serialize.
    path :
        Output YAML path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
        )


def get_nested(cfg: Dict[str, Any], *keys: str, default=None):
    """
    Safely retrieve a nested value from a dictionary.

    Parameters
    ----------
    cfg :
        Source dictionary.
    *keys :
        Sequence of nested keys.
    default :
        Value returned if any key is missing.

    Returns
    -------
    Any
        Nested value if present, otherwise default.
    """
    out = cfg

    for key in keys:
        if not isinstance(out, dict) or key not in out:
            return default
        out = out[key]

    return out


# ---------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------

def resolve_from_training_config(
    config_path: str | Path,
    split: str,
) -> Dict[str, Any]:
    """
    Resolve datasets arguments from a training YAML config.

    Required YAML sections:

        model.type
        datasets.*

    The output dictionary has the same fields used by the legacy CLI mode.

    Parameters
    ----------
    config_path :
        Path to the training YAML config.
    split :
        Dataset split to build. Must be "train", "val", or "test".

    Returns
    -------
    dict
        Resolved datasets arguments.
    """
    cfg = load_yaml(config_path)

    model_name = get_nested(cfg, "data", "model_name", default=None)
    if model_name is None:
        model_name = get_nested(cfg, "model", "type", default=None)

    if model_name is None:
        raise ValueError(
            "Cannot infer model_name from YAML. Expected either "
            "`data.model_name` or `model.type`."
        )

    pp = cfg.get("datasets", None)
    if pp is None:
        raise ValueError(
            f"Missing `datasets` section in config: {config_path}"
        )

    if split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split: {split}")

    raw_key = f"raw_{split}_samples"
    processed_key = f"processed_{split}_samples"

    if raw_key not in pp:
        raise ValueError(
            f"Missing datasets.{raw_key} in config: {config_path}"
        )

    if processed_key not in pp:
        raise ValueError(
            f"Missing datasets.{processed_key} in config: {config_path}"
        )

    resolved = {
        "samples": pp[raw_key],
        "model_name": model_name,
        "split": split,
        "out_root": pp.get("out_root", "data/processed"),
        "out_samples": pp[processed_key],
        "patch_size_px": int(pp.get("patch_size_px", 512)),
        "stride_px": pp.get("stride_px", None),
        "patches_per_image": pp.get("patches_per_image", "auto"),
        "patches_per_image_by_sensor": pp.get("patches_per_image_by_sensor", {}),
        "max_patches_per_image": int(pp.get("max_patches_per_image", 20)),
        "max_retries": int(pp.get("max_retries", 50)),
        "max_invalid_frac": float(pp.get("max_invalid_frac", 0.9)),
        "min_valid_frac": float(pp.get("min_valid_frac", 0.1)),
        "min_valid_classes": int(pp.get("min_valid_classes", 1)),
        "seed": int(pp.get("seed", 42)),
        "skip_existing": bool(pp.get("skip_existing", False)),
        "stop_on_error": bool(pp.get("stop_on_error", False)),
    }

    if resolved["stride_px"] is not None:
        resolved["stride_px"] = int(resolved["stride_px"])

    return resolved


def resolve_from_cli(args) -> Dict[str, Any]:
    """
    Resolve datasets arguments from explicit CLI arguments.

    Parameters
    ----------
    args :
        Parsed command-line arguments.

    Returns
    -------
    dict
        Resolved datasets arguments.

    Raises
    ------
    ValueError
        If required legacy CLI arguments are missing.
    """
    required = {
        "--samples": args.samples,
        "--model-name": args.model_name,
        "--out-samples": args.out_samples,
    }

    missing = [name for name, value in required.items() if value is None]

    if missing:
        raise ValueError(
            "Missing required arguments in legacy CLI mode: "
            f"{missing}. Either provide them or use `--config`."
        )

    return {
        "samples": args.samples,
        "model_name": args.model_name,
        "split": args.split,
        "out_root": args.out_root,
        "out_samples": args.out_samples,
        "patch_size_px": args.patch_size_px,
        "stride_px": args.stride_px,
        "patches_per_image": args.patches_per_image,
        "max_retries": args.max_retries,
        "max_invalid_frac": args.max_invalid_frac,
        "min_valid_frac": args.min_valid_frac,
        "min_valid_classes": args.min_valid_classes,
        "seed": args.seed,
        "skip_existing": args.skip_existing,
        "stop_on_error": args.stop_on_error,
    }


# ---------------------------------------------------------------------
# Raster utilities
# ---------------------------------------------------------------------

def read_raster_profile(path: str | Path) -> Dict[str, Any]:
    """
    Read the Rasterio profile of the source GeoTIFF.

    Parameters
    ----------
    path :
        Path to the source raster.

    Returns
    -------
    dict
        Rasterio profile dictionary.
    """
    with rasterio.open(path) as src:
        return src.profile.copy()


def update_transform_after_crop_and_resize(
    src_transform,
    row: int,
    col: int,
    crop_px: int,
    out_height: int,
    out_width: int,
):
    """
    Compute the transform of the preprocessed patch.

    Steps
    -----
    1. Translate the origin to the crop position.
    2. Scale the pixel size if the crop is resized.

    Example
    -------
        raw crop 512 x 512
        output   504 x 504

    The processed GeoTIFF preserves the geographic extent of the raw crop.

    Parameters
    ----------
    src_transform :
        Source raster affine transform.
    row :
        Crop top-left row.
    col :
        Crop top-left column.
    crop_px :
        Raw crop size in pixels.
    out_height :
        Output patch height.
    out_width :
        Output patch width.

    Returns
    -------
    affine.Affine
        Updated affine transform for the processed patch.
    """
    crop_transform = src_transform * Affine.translation(col, row)

    scale_x = crop_px / out_width
    scale_y = crop_px / out_height

    return crop_transform * Affine.scale(scale_x, scale_y)


def save_image_geotiff(
    path: str | Path,
    image: np.ndarray,
    source_profile: Dict[str, Any],
    transform,
) -> None:
    """
    Save a preprocessed image as a multiband float32 GeoTIFF.

    Parameters
    ----------
    path :
        Output image GeoTIFF path.
    image :
        Image array with shape (C, H, W).
    source_profile :
        Rasterio profile copied from the source image.
    transform :
        Affine transform of the processed patch.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if image.ndim != 3:
        raise ValueError(f"image must have shape (C, H, W), got {image.shape}")

    c, h, w = image.shape

    profile = source_profile.copy()
    profile.update(
        {
            "driver": "GTiff",
            "count": c,
            "height": h,
            "width": w,
            "dtype": "float32",
            "nodata": None,
            "transform": transform,
            "compress": "lzw",
        }
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(image.astype(np.float32))


def save_label_geotiff(
    path: str | Path,
    label: np.ndarray,
    source_profile: Dict[str, Any],
    transform,
    ignore_index: int = IGNORE_INDEX,
) -> None:
    """
    Save a preprocessed label as a single-band uint8 GeoTIFF.

    Parameters
    ----------
    path :
        Output label GeoTIFF path.
    label :
        Label array with shape (H, W).
    source_profile :
        Rasterio profile copied from the source image.
    transform :
        Affine transform of the processed patch.
    ignore_index :
        Value used as nodata in the output label GeoTIFF.

    Values
    ------
        0   = sealed_soil
        1   = non_sealed_soil
        255 = ignore_index
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if label.ndim != 2:
        raise ValueError(f"label must have shape (H, W), got {label.shape}")

    h, w = label.shape

    profile = source_profile.copy()
    profile.update(
        {
            "driver": "GTiff",
            "count": 1,
            "height": h,
            "width": w,
            "dtype": "uint8",
            "nodata": int(ignore_index),
            "transform": transform,
            "compress": "lzw",
        }
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(label.astype(np.uint8), 1)


# ---------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------

def build_output_paths(
    out_root: Path,
    model_name: str,
    sensor: str,
    split: str,
    stem: str,
    patch_idx: int,
) -> Tuple[Path, Path]:
    """
    Build output image and label paths for one processed patch.

    Parameters
    ----------
    out_root :
        Root directory of the processed datasets.
    model_name :
        Target model name.
    sensor :
        Sensor name.
    split :
        Dataset split.
    stem :
        Source image stem.
    patch_idx :
        Local patch index for the current source image.

    Returns
    -------
    tuple of pathlib.Path
        Output image path and output label path.
    """
    base = out_root / model_name / sensor / split

    image_dir = base / "images"
    label_dir = base / "labels"

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    name = f"{stem}_patch_{patch_idx:06d}.tif"

    return image_dir / name, label_dir / name


# ---------------------------------------------------------------------
# Preprocessing patch
# ---------------------------------------------------------------------

def process_patch(
    sample: Dict[str, Any],
    info,
    model_name: str,
    item: Dict[str, int],
):
    """
    Read, preprocess and return one image/label patch.

    Parameters
    ----------
    sample :
        Raw sample dictionary containing image and label paths.
    info :
        Sensor configuration associated with the sample.
    model_name :
        Target model name.
    item :
        Crop item containing row, col and crop_px.

    Returns
    -------
    tuple
        image_proc :
            Preprocessed image array.
        label_proc :
            Preprocessed label array.
        wavelengths :
            Wavelengths of the selected image bands.
    """
    image_raw = read_image_window(
        path=sample["image"],
        row=item["row"],
        col=item["col"],
        size=item["crop_px"],
    )

    label_raw = read_label_window(
        path=sample["label"],
        row=item["row"],
        col=item["col"],
        size=item["crop_px"],
    )

    image_proc, wavelengths = preprocess_image_for_model(
        x=image_raw,
        info=info,
        model_name=model_name,
    )

    label_proc = preprocess_label_for_model(
        y=label_raw,
        info=info,
        model_name=model_name,
        remap=True,
        ignore_index=IGNORE_INDEX,
    )

    return image_proc, label_proc, wavelengths


def estimate_auto_patches_from_raster(
    h: int,
    w: int,
    crop_px: int,
    max_patches_per_image: int,
) -> int:
    """
    Estimate the number of random training patches from the actual raster size.

    This is a safety fallback.

    Examples
    --------
        512 x 512   crop 512 -> 1
        1024 x 1024 crop 512 -> 4
        2048 x 2048 crop 512 -> 16

    Parameters
    ----------
    h :
        Raster height.
    w :
        Raster width.
    crop_px :
        Crop size in pixels.
    max_patches_per_image :
        Upper limit for the number of patches per image.

    Returns
    -------
    int
        Estimated number of random patches.
    """
    import math

    n_h = max(1, math.ceil(h / crop_px))
    n_w = max(1, math.ceil(w / crop_px))

    n = n_h * n_w

    return max(1, min(int(n), int(max_patches_per_image)))


def resolve_patches_per_image_for_sample(
    *,
    sensor: str,
    h: int,
    w: int,
    crop_px: int,
    patches_per_image,
    patches_per_image_by_sensor: Dict[str, int],
    max_patches_per_image: int,
) -> int:
    """
    Resolve patches_per_image for a specific sample.

    Priority
    --------
    1. patches_per_image_by_sensor[sensor]
    2. integer patches_per_image
    3. automatic estimate from raster size

    Parameters
    ----------
    sensor :
        Sensor name.
    h :
        Raster height.
    w :
        Raster width.
    crop_px :
        Crop size in pixels.
    patches_per_image :
        Global patches_per_image setting. Can be an int or "auto".
    patches_per_image_by_sensor :
        Optional sensor-specific override dictionary.
    max_patches_per_image :
        Maximum number of patches per image when using automatic estimation.

    Returns
    -------
    int
        Number of random patches to sample for this image.
    """
    if sensor in patches_per_image_by_sensor:
        return int(patches_per_image_by_sensor[sensor])

    if isinstance(patches_per_image, int):
        return int(patches_per_image)

    if isinstance(patches_per_image, str):
        if patches_per_image.lower() == "auto":
            return estimate_auto_patches_from_raster(
                h=h,
                w=w,
                crop_px=crop_px,
                max_patches_per_image=max_patches_per_image,
            )

        try:
            return int(patches_per_image)
        except ValueError:
            pass

    return estimate_auto_patches_from_raster(
        h=h,
        w=w,
        crop_px=crop_px,
        max_patches_per_image=max_patches_per_image,
    )


# ---------------------------------------------------------------------
# Build datasets
# ---------------------------------------------------------------------

def build_processed_dataset(resolved: Dict[str, Any]) -> None:
    """
    Build the offline preprocessed GeoTIFF datasets.

    Parameters
    ----------
    resolved :
        Dictionary containing all resolved datasets arguments.
    """
    samples_path = resolved["samples"]
    model_name = resolved["model_name"]
    split = resolved["split"]
    out_root = Path(resolved["out_root"])
    out_samples_path = Path(resolved["out_samples"])

    patch_size_px = int(resolved["patch_size_px"])
    stride_px = resolved["stride_px"]
    patches_per_image = resolved["patches_per_image"]
    patches_per_image_by_sensor = resolved.get("patches_per_image_by_sensor", {})
    max_patches_per_image = int(resolved.get("max_patches_per_image", 20))
    max_retries = int(resolved["max_retries"])

    max_invalid_frac = float(resolved["max_invalid_frac"])
    min_valid_frac = float(resolved["min_valid_frac"])
    min_valid_classes = int(resolved["min_valid_classes"])

    seed = int(resolved["seed"])
    skip_existing = bool(resolved["skip_existing"])
    stop_on_error = bool(resolved["stop_on_error"])

    samples = load_yaml(samples_path)

    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError(f"Split YAML is empty or invalid: {samples_path}")

    rng = random.Random(seed)

    processed_samples = []
    errors = []

    print("=" * 80)
    print("BUILD PROCESSED GEOTIFF DATASET")
    print("=" * 80)
    print("Raw samples       :", samples_path)
    print("Model name        :", model_name)
    print("Split             :", split)
    print("Out root          :", out_root)
    print("Out samples       :", out_samples_path)
    print("Patch size px     :", patch_size_px)
    print("Stride px         :", stride_px if stride_px is not None else "auto")
    print("Patches per image :", patches_per_image if split == "train" else "grid")
    print("Seed              :", seed)
    print("Skip existing     :", skip_existing)
    print("Output format     : GeoTIFF")
    print("=" * 80)

    for sample_idx, sample in enumerate(tqdm(samples, desc=f"Processing {split}")):
        try:
            sensor = sample.get("sensor", "unknown")
            sensor_config = infer_sensor_config_path(sample)

            info = read_sensor(
                image_path=sample["image"],
                sensor_config_path=sensor_config,
            )

            source_profile = read_raster_profile(sample["image"])
            source_transform = source_profile["transform"]

            h, w = raster_size(sample["image"])
            crop_px = min(patch_size_px, h, w)

            if split == "train":
                sample_patches_per_image = resolve_patches_per_image_for_sample(
                    sensor=sensor,
                    h=h,
                    w=w,
                    crop_px=crop_px,
                    patches_per_image=patches_per_image,
                    patches_per_image_by_sensor=patches_per_image_by_sensor,
                    max_patches_per_image=max_patches_per_image,
                )

                items = sample_random_valid_windows(
                    sample=sample,
                    info=info,
                    crop_px=crop_px,
                    patches_per_image=sample_patches_per_image,
                    max_retries=max_retries,
                    max_invalid_frac=max_invalid_frac,
                    min_valid_frac=min_valid_frac,
                    min_valid_classes=min_valid_classes,
                    rng=rng,
                )
            else:
                effective_stride_px = stride_px if stride_px is not None else crop_px

                items = make_grid_items(
                    h=h,
                    w=w,
                    crop_px=crop_px,
                    stride_px=int(effective_stride_px),
                )

            if len(items) == 0:
                print(
                    f"\n[WARNING] No valid patch found for sample_idx={sample_idx}, "
                    f"image={sample.get('image')}"
                )
                continue

            stem = Path(sample["image"]).stem

            for local_patch_idx, item in enumerate(items):
                out_image, out_label = build_output_paths(
                    out_root=out_root,
                    model_name=model_name,
                    sensor=sensor,
                    split=split,
                    stem=stem,
                    patch_idx=local_patch_idx,
                )

                if skip_existing and out_image.exists() and out_label.exists():
                    processed_samples.append(
                        {
                            "image": out_image.as_posix(),
                            "label": out_label.as_posix(),
                            "sensor": sensor,
                            "model_name": model_name,
                            "source_image": sample["image"],
                            "source_label": sample["label"],
                            "row": item["row"],
                            "col": item["col"],
                            "crop_px": item["crop_px"],
                        }
                    )
                    continue

                image_proc, label_proc, wavelengths = process_patch(
                    sample=sample,
                    info=info,
                    model_name=model_name,
                    item=item,
                )

                out_h, out_w = label_proc.shape

                patch_transform = update_transform_after_crop_and_resize(
                    src_transform=source_transform,
                    row=item["row"],
                    col=item["col"],
                    crop_px=item["crop_px"],
                    out_height=out_h,
                    out_width=out_w,
                )

                save_image_geotiff(
                    path=out_image,
                    image=image_proc,
                    source_profile=source_profile,
                    transform=patch_transform,
                )

                save_label_geotiff(
                    path=out_label,
                    label=label_proc,
                    source_profile=source_profile,
                    transform=patch_transform,
                    ignore_index=IGNORE_INDEX,
                )

                processed_samples.append(
                    {
                        "image": out_image.as_posix(),
                        "label": out_label.as_posix(),
                        "sensor": sensor,
                        "model_name": model_name,
                        "source_image": sample["image"],
                        "source_label": sample["label"],
                        "row": item["row"],
                        "col": item["col"],
                        "crop_px": item["crop_px"],
                        "wavelengths": [float(w) for w in wavelengths],
                    }
                )

        except Exception as e:
            err = {
                "sample_idx": sample_idx,
                "image": sample.get("image"),
                "label": sample.get("label"),
                "error": str(e),
            }
            errors.append(err)

            print(f"\n[ERROR] sample_idx={sample_idx}: {e}")

            if stop_on_error:
                raise

    save_yaml(processed_samples, out_samples_path)

    summary_path = out_samples_path.with_name(
        out_samples_path.stem + "_summary.yaml"
    )

    save_yaml(
        {
            "raw_samples": str(samples_path),
            "processed_samples": out_samples_path.as_posix(),
            "model_name": model_name,
            "split": split,
            "output_format": "geotiff",
            "patch_size_px": patch_size_px,
            "stride_px": stride_px,
            "patches_per_image": patches_per_image,
            "max_retries": max_retries,
            "max_invalid_frac": max_invalid_frac,
            "min_valid_frac": min_valid_frac,
            "min_valid_classes": min_valid_classes,
            "seed": seed,
            "skip_existing": skip_existing,
            "num_processed_samples": len(processed_samples),
            "num_errors": len(errors),
            "errors": errors,
            "patches_per_image_by_sensor": patches_per_image_by_sensor,
            "max_patches_per_image": max_patches_per_image,
        },
        summary_path,
    )

    print("=" * 80)
    print("DONE")
    print("Processed samples :", len(processed_samples))
    print("Errors            :", len(errors))
    print("Samples YAML      :", out_samples_path)
    print("Summary           :", summary_path)
    print("=" * 80)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser()

    # New recommended mode
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Training YAML config containing a `datasets` section. "
            "Recommended mode."
        ),
    )

    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "val", "test"],
        help="Dataset split to build.",
    )

    # Legacy explicit mode
    parser.add_argument("--samples", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--out-root", default="data/processed")
    parser.add_argument("--out-samples", default=None)

    parser.add_argument("--patch-size-px", type=int, default=512)
    parser.add_argument("--stride-px", type=int, default=None)

    parser.add_argument("--patches-per-image", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=50)

    parser.add_argument("--max-invalid-frac", type=float, default=0.9)
    parser.add_argument("--min-valid-frac", type=float, default=0.1)
    parser.add_argument("--min-valid-classes", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")

    # Optional runtime overrides when using --config
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "When using --config, this disables skip_existing. "
            "Equivalent to rebuilding existing files."
        ),
    )

    parser.add_argument(
        "--override-patches-per-image",
        type=int,
        default=None,
        help="Optional override for datasets.patches_per_image.",
    )

    parser.add_argument(
        "--override-patch-size-px",
        type=int,
        default=None,
        help="Optional override for datasets.patch_size_px.",
    )

    parser.add_argument(
        "--override-stride-px",
        type=int,
        default=None,
        help="Optional override for datasets.stride_px.",
    )

    return parser.parse_args()


def main() -> None:
    """
    CLI entry point.

    The function resolves datasets arguments either from a training config
    YAML or from legacy explicit CLI arguments, then builds the processed
    GeoTIFF datasets.
    """
    args = parse_args()

    if args.config is not None:
        resolved = resolve_from_training_config(
            config_path=args.config,
            split=args.split,
        )

        # Runtime overrides
        if args.overwrite:
            resolved["skip_existing"] = False

        if args.override_patches_per_image is not None:
            resolved["patches_per_image"] = int(args.override_patches_per_image)

        if args.override_patch_size_px is not None:
            resolved["patch_size_px"] = int(args.override_patch_size_px)

        if args.override_stride_px is not None:
            resolved["stride_px"] = int(args.override_stride_px)

    else:
        resolved = resolve_from_cli(args)

    build_processed_dataset(resolved)


if __name__ == "__main__":
    main()