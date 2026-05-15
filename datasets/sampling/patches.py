"""
preprocessing/patches.py
------------------------

Shared raster window and crop utilities.

This module contains only generic functions used by both:

  - online datasets;
  - offline processed-datasets builders.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window

from datasets.preprocessing.pipeline import is_valid_patch


def raster_size(path: str | Path) -> Tuple[int, int]:
    """
    Return raster spatial size as (height, width).
    """
    with rasterio.open(path) as src:
        return src.height, src.width


def read_image_window(
    path: str | Path,
    row: int,
    col: int,
    size: int,
) -> np.ndarray:
    """
    Read a square window from a multi-band raster.

    Returns
    -------
    np.ndarray
        Array with shape (C, H, W).
    """
    with rasterio.open(path) as src:
        window = Window(
            col_off=col,
            row_off=row,
            width=size,
            height=size,
        )

        window = window.intersection(
            Window(0, 0, src.width, src.height)
        )

        return src.read(window=window)


def read_label_window(
    path: str | Path,
    row: int,
    col: int,
    size: int,
) -> np.ndarray:
    """
    Read a square window from a single-band label raster.

    Returns
    -------
    np.ndarray
        Array with shape (H, W), dtype int64.
    """
    with rasterio.open(path) as src:
        window = Window(
            col_off=col,
            row_off=row,
            width=size,
            height=size,
        )

        window = window.intersection(
            Window(0, 0, src.width, src.height)
        )

        return src.read(1, window=window).astype(np.int64)


def compute_crop_size_px(
    image_path: str | Path,
    sensor_info: Any,
    patch_size_px: Optional[int],
    patch_size_m: Optional[float],
) -> int:
    """
    Determine crop size in pixels.

    Priority:
      1. patch_size_px
      2. patch_size_m / sensor_info.gsd_m
      3. min(height, width)
    """
    if patch_size_px is not None:
        if patch_size_px <= 0:
            raise ValueError(
                f"patch_size_px must be > 0, received {patch_size_px}"
            )
        return int(patch_size_px)

    if patch_size_m is not None:
        gsd_m = getattr(sensor_info, "gsd_m", None)

        if gsd_m is None:
            raise ValueError(
                "patch_size_m was provided, but SensorConfig does not contain gsd_m."
            )

        if float(gsd_m) <= 0:
            raise ValueError(f"Invalid gsd_m: {gsd_m}")

        crop_px = int(round(float(patch_size_m) / float(gsd_m)))

        if crop_px <= 0:
            raise ValueError(
                f"Invalid crop_px: patch_size_m={patch_size_m}, gsd_m={gsd_m}"
            )

        return crop_px

    h, w = raster_size(image_path)
    return int(min(h, w))


def make_grid_items(
    h: int,
    w: int,
    crop_px: int,
    stride_px: int,
    sample_idx: Optional[int] = None,
) -> List[Dict[str, int]]:
    """
    Build regular grid crop items.

    If sample_idx is provided, each item includes it.
    """
    if stride_px <= 0:
        raise ValueError(f"stride_px must be > 0, received {stride_px}")

    crop_px = int(min(crop_px, h, w))

    max_row = max(0, h - crop_px)
    max_col = max(0, w - crop_px)

    rows = list(range(0, max_row + 1, stride_px))
    cols = list(range(0, max_col + 1, stride_px))

    if not rows:
        rows = [0]

    if not cols:
        cols = [0]

    if rows[-1] != max_row:
        rows.append(max_row)

    if cols[-1] != max_col:
        cols.append(max_col)

    items: List[Dict[str, int]] = []

    for row in rows:
        for col in cols:
            item = {
                "row": int(row),
                "col": int(col),
                "crop_px": int(crop_px),
            }

            if sample_idx is not None:
                item["sample_idx"] = int(sample_idx)

            items.append(item)

    return items


def sample_random_valid_window(
    *,
    sample: Dict[str, Any],
    info: Any,
    crop_px: int,
    max_retries: int,
    max_invalid_frac: float,
    min_valid_frac: float,
    min_valid_classes: int,
    rng: random.Random | None = None,
    sample_idx: Optional[int] = None,
) -> Dict[str, int]:
    """
    Sample one random valid crop window.

    If no valid patch is found within max_retries, the last sampled window is
    returned.
    """
    rng = rng if rng is not None else random

    h, w = raster_size(sample["image"])
    crop_px = int(min(crop_px, h, w))

    last_item = {
        "row": 0,
        "col": 0,
        "crop_px": int(crop_px),
    }

    if sample_idx is not None:
        last_item["sample_idx"] = int(sample_idx)

    for _ in range(max_retries):
        row = rng.randint(0, max(0, h - crop_px))
        col = rng.randint(0, max(0, w - crop_px))

        label_raw = read_label_window(
            path=sample["label"],
            row=row,
            col=col,
            size=crop_px,
        )

        last_item = {
            "row": int(row),
            "col": int(col),
            "crop_px": int(crop_px),
        }

        if sample_idx is not None:
            last_item["sample_idx"] = int(sample_idx)

        if is_valid_patch(
            label=label_raw,
            info=info,
            max_invalid_frac=max_invalid_frac,
            min_valid_frac=min_valid_frac,
            min_valid_classes=min_valid_classes,
        ):
            return last_item

    return last_item


def sample_random_valid_windows(
    *,
    sample: Dict[str, Any],
    info: Any,
    crop_px: int,
    patches_per_image: int,
    max_retries: int,
    max_invalid_frac: float,
    min_valid_frac: float,
    min_valid_classes: int,
    rng: random.Random,
) -> List[Dict[str, int]]:
    """
    Sample multiple valid random crop windows from one image.

    This is mainly used by offline datasets builders.
    """
    h, w = raster_size(sample["image"])
    crop_px = int(min(crop_px, h, w))

    items: List[Dict[str, int]] = []

    attempts_total = int(patches_per_image) * int(max_retries)
    attempts = 0

    while len(items) < patches_per_image and attempts < attempts_total:
        attempts += 1

        row = rng.randint(0, max(0, h - crop_px))
        col = rng.randint(0, max(0, w - crop_px))

        label_raw = read_label_window(
            path=sample["label"],
            row=row,
            col=col,
            size=crop_px,
        )

        ok = is_valid_patch(
            label=label_raw,
            info=info,
            max_invalid_frac=max_invalid_frac,
            min_valid_frac=min_valid_frac,
            min_valid_classes=min_valid_classes,
        )

        if not ok:
            continue

        items.append(
            {
                "row": int(row),
                "col": int(col),
                "crop_px": int(crop_px),
            }
        )

    return items