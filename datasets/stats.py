"""
datasets/stats.py
----------------------

Dataset statistics utilities for remote-sensing data.

This module is used to:

  - compute per-channel mean/std;
  - compute class distribution;
  - estimate the invalid-pixel fraction;
  - debug datasets content.

It should preferably be used only on the training set.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from tqdm import tqdm

from datasets.preprocessing.pipeline import IGNORE_INDEX


@torch.no_grad()
def compute_image_mean_std(
    dataloader,
    max_batches: Optional[int] = None,
) -> Dict:
    """
    Compute per-channel mean and standard deviation on preprocessed images.

    Expected dataloader input
    -------------------------
        batch["image"] : Tensor (B, C, H, W)

    Parameters
    ----------
    dataloader :
        PyTorch dataloader returning batches with an "image" tensor.
    max_batches :
        Optional maximum number of batches to process.

    Returns
    -------
    dict
        Dictionary containing:

        mean :
            Per-channel mean values.
        std :
            Per-channel standard deviation values.
        num_pixels :
            Number of pixels processed per channel.
        num_batches :
            Number of batches processed.
    """
    channel_sum = None
    channel_sum_sq = None
    num_pixels = 0
    num_batches = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing mean/std")):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = batch["image"].float()  # (B, C, H, W)

        if x.ndim != 4:
            raise ValueError(
                f"batch['image'] must have shape (B, C, H, W), received {x.shape}"
            )

        b, c, h, w = x.shape

        if channel_sum is None:
            channel_sum = torch.zeros(c, dtype=torch.float64)
            channel_sum_sq = torch.zeros(c, dtype=torch.float64)

        if c != channel_sum.numel():
            raise ValueError(
                f"Variable number of channels in dataloader: first C={channel_sum.numel()}, "
                f"now C={c}. Compute stats separately by model/sensor or use homogeneous batches."
            )

        x64 = x.double()

        channel_sum += x64.sum(dim=(0, 2, 3))
        channel_sum_sq += (x64 ** 2).sum(dim=(0, 2, 3))

        num_pixels += b * h * w
        num_batches += 1

    if num_pixels == 0:
        raise RuntimeError("No pixels processed. Check the dataloader.")

    mean = channel_sum / num_pixels
    var = channel_sum_sq / num_pixels - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=0.0))

    return {
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "num_pixels": int(num_pixels),
        "num_batches": int(num_batches),
    }


@torch.no_grad()
def compute_label_distribution(
    dataloader,
    ignore_index: int = IGNORE_INDEX,
    num_classes: int = 2,
    max_batches: Optional[int] = None,
) -> Dict:
    """
    Compute class distribution on preprocessed labels.

    Expected labels
    ---------------
        0   = sealed_soil
        1   = non_sealed_soil
        255 = ignore_index

    Parameters
    ----------
    dataloader :
        PyTorch dataloader returning batches with a "label" tensor.
    ignore_index :
        Label value ignored during training/evaluation.
    num_classes :
        Number of semantic classes, excluding ignore_index.
    max_batches :
        Optional maximum number of batches to process.

    Returns
    -------
    dict
        Dictionary containing:

        counts :
            Pixel counts for each class and ignore_index.
        fractions_on_valid :
            Class fractions computed only on valid pixels.
        ignore_fraction :
            Fraction of ignored pixels over all pixels.
        valid_fraction :
            Fraction of valid pixels over all pixels.
        num_batches :
            Number of batches processed.
    """
    class_counts = torch.zeros(num_classes, dtype=torch.long)
    ignore_count = 0
    total_count = 0
    num_batches = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing label stats")):
        if max_batches is not None and batch_idx >= max_batches:
            break

        y = batch["label"].long()

        if y.ndim != 3:
            raise ValueError(
                f"batch['label'] must have shape (B, H, W), received {y.shape}"
            )

        total_count += y.numel()
        ignore_count += int((y == ignore_index).sum().item())

        valid_mask = y != ignore_index
        y_valid = y[valid_mask]

        for cls in range(num_classes):
            class_counts[cls] += int((y_valid == cls).sum().item())

        num_batches += 1

    valid_count = int(class_counts.sum().item())

    if total_count == 0:
        raise RuntimeError("No pixels processed.")

    fractions_on_valid = {}

    for cls in range(num_classes):
        if valid_count > 0:
            fractions_on_valid[str(cls)] = float(class_counts[cls].item()) / float(valid_count)
        else:
            fractions_on_valid[str(cls)] = 0.0

    counts = {str(cls): int(class_counts[cls].item()) for cls in range(num_classes)}
    counts["ignore"] = int(ignore_count)

    return {
        "counts": counts,
        "fractions_on_valid": fractions_on_valid,
        "ignore_fraction": float(ignore_count) / float(total_count),
        "valid_fraction": float(valid_count) / float(total_count),
        "num_batches": int(num_batches),
    }


@torch.no_grad()
def compute_stats_by_sensor(
    dataloader,
    ignore_index: int = IGNORE_INDEX,
    num_classes: int = 2,
    max_batches: Optional[int] = None,
) -> Dict:
    """
    Compute label statistics separately for each sensor.

    Expected dataloader input
    -------------------------
        batch["sensor"] : list[str]
        batch["label"]  : Tensor (B, H, W)

    Parameters
    ----------
    dataloader :
        PyTorch dataloader returning batches with "sensor" and "label".
    ignore_index :
        Label value ignored during training/evaluation.
    num_classes :
        Number of semantic classes, excluding ignore_index.
    max_batches :
        Optional maximum number of batches to process.

    Returns
    -------
    dict
        Dictionary indexed by sensor name. For each sensor, it contains:

        counts :
            Pixel counts for each class and ignore_index.
        fractions_on_valid :
            Class fractions computed only on valid pixels.
        ignore_fraction :
            Fraction of ignored pixels over all pixels.
        valid_fraction :
            Fraction of valid pixels over all pixels.
    """
    stats = {}

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing stats by sensor")):
        if max_batches is not None and batch_idx >= max_batches:
            break

        sensors = batch["sensor"]
        labels = batch["label"].long()

        if labels.ndim != 3:
            raise ValueError(
                f"batch['label'] must have shape (B, H, W), received {labels.shape}"
            )

        if len(sensors) != labels.shape[0]:
            raise ValueError(
                f"Number of sensors and labels does not match: "
                f"len(sensors)={len(sensors)}, batch_size={labels.shape[0]}"
            )

        for i, sensor in enumerate(sensors):
            if sensor not in stats:
                stats[sensor] = {
                    "class_counts": torch.zeros(num_classes, dtype=torch.long),
                    "ignore_count": 0,
                    "total_count": 0,
                }

            y = labels[i]

            stats[sensor]["total_count"] += y.numel()
            stats[sensor]["ignore_count"] += int((y == ignore_index).sum().item())

            valid = y[y != ignore_index]

            for cls in range(num_classes):
                stats[sensor]["class_counts"][cls] += int((valid == cls).sum().item())

    out = {}

    for sensor, s in stats.items():
        class_counts = s["class_counts"]
        valid_count = int(class_counts.sum().item())
        total_count = int(s["total_count"])
        ignore_count = int(s["ignore_count"])

        fractions = {}

        for cls in range(num_classes):
            fractions[str(cls)] = (
                float(class_counts[cls].item()) / float(valid_count)
                if valid_count > 0 else 0.0
            )

        out[sensor] = {
            "counts": {
                **{str(cls): int(class_counts[cls].item()) for cls in range(num_classes)},
                "ignore": ignore_count,
            },
            "fractions_on_valid": fractions,
            "ignore_fraction": ignore_count / total_count if total_count > 0 else 0.0,
            "valid_fraction": valid_count / total_count if total_count > 0 else 0.0,
        }

    return out