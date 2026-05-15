"""
scripts/inspect_dataset.py
------------------------

Check that the datasets works correctly before training.

This script uses the real pipeline:

    split YAML
        ↓
    MultiSensorSegDataset
        ↓
    preprocess_image_for_model()
    preprocess_label_for_model()
        ↓
    DataLoader + collate
        ↓
    batch ready for the model

No model is trained.

Examples
--------

Check SPOT train split:

    python scripts/inspect_dataset.py --samples configs/splits/spot_train_samples.yaml --model-name crossearth --split train --patch-size-px 512 --batch-size 4 --num-workers 0

Check SPOT validation split:

    python scripts/inspect_dataset.py --samples configs/splits/spot_val_samples.yaml --model-name crossearth --split val --patch-size-px 512 --stride-px 512 --batch-size 4 --num-workers 0

Check mixed SPOT + PlanetScope split:

    python scripts/inspect_dataset.py --samples configs/splits/mixed_train_samples.yaml --model-name crossearth --split train --patch-size-px 512 --batch-size 4 --sensor-batch-sampler --num-workers 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import yaml
from torch.utils.data import DataLoader

from datasets.core.raw import MultiSensorSegDataset
from datasets.collate import (
    segmentation_collate,
    SensorBatchSampler,
)
from datasets.transforms.augment import (
    SegmentationEvalTransform,
    SegmentationTrainTransform,
)
from datasets.preprocessing.pipeline import IGNORE_INDEX


# ---------------------------------------------------------------------
# Utilities
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
        raise FileNotFoundError(f"YAML file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def summarize_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize the list of samples loaded from a split YAML.

    Parameters
    ----------
    samples :
        List of sample dictionaries.

    Returns
    -------
    dict
        Summary containing number of samples, counts by sensor and first sample.
    """
    by_sensor: Dict[str, int] = {}

    for sample in samples:
        sensor = sample.get("sensor", "unknown")
        by_sensor[sensor] = by_sensor.get(sensor, 0) + 1

    return {
        "num_samples": len(samples),
        "by_sensor": by_sensor,
        "first_sample": samples[0] if samples else None,
    }


def count_labels_in_batch(labels: torch.Tensor) -> Dict[int, int]:
    """
    Count label values in a batch.

    Parameters
    ----------
    labels :
        Label tensor with shape (B, H, W).

    Returns
    -------
    dict
        Mapping from label value to pixel count.
    """
    values, counts = torch.unique(labels.cpu(), return_counts=True)

    return {
        int(v): int(c)
        for v, c in zip(values.tolist(), counts.tolist())
    }


def label_name(value: int) -> str:
    """
    Convert a numeric label value to a human-readable label name.

    Parameters
    ----------
    value :
        Numeric label value.

    Returns
    -------
    str
        Human-readable label name.
    """
    if value == 0:
        return "sealed_soil"
    if value == 1:
        return "non_sealed_soil"
    if value == IGNORE_INDEX:
        return "ignore_index"
    return "unknown"


def print_label_counts(counts: Dict[int, int]) -> None:
    """
    Print label counts and fractions in a readable format.

    Parameters
    ----------
    counts :
        Mapping from label value to pixel count.
    """
    total = sum(counts.values())

    for value in sorted(counts):
        frac = counts[value] / total if total > 0 else 0.0
        print(
            f"    {value:>3} ({label_name(value):>15}) : "
            f"{counts[value]:>10} px  ({frac:.4%})"
        )


def validate_batch(
    batch: Dict[str, Any],
    expected_channels: int | None,
    expected_size: int | None,
) -> None:
    """
    Run minimum consistency checks on one batch.

    Parameters
    ----------
    batch :
        Batch dictionary returned by the DataLoader.
    expected_channels :
        Expected number of image channels. If None, the check is skipped.
    expected_size :
        Expected spatial size. If None, the check is skipped.

    Raises
    ------
    ValueError
        If the batch is missing required keys or has inconsistent shapes.
    """
    if "image" not in batch:
        raise ValueError("Batch does not contain key 'image'.")

    if "label" not in batch:
        raise ValueError("Batch does not contain key 'label'.")

    images = batch["image"]
    labels = batch["label"]

    if images.ndim != 4:
        raise ValueError(
            f"batch['image'] must have shape (B, C, H, W), received {images.shape}"
        )

    if labels.ndim != 3:
        raise ValueError(
            f"batch['label'] must have shape (B, H, W), received {labels.shape}"
        )

    b, c, h, w = images.shape

    if labels.shape[0] != b:
        raise ValueError(
            f"Inconsistent batch size: image B={b}, label B={labels.shape[0]}"
        )

    if labels.shape[-2:] != images.shape[-2:]:
        raise ValueError(
            f"Inconsistent spatial shapes: image={images.shape}, label={labels.shape}"
        )

    if expected_channels is not None and c != expected_channels:
        raise ValueError(
            f"Unexpected channels: expected C={expected_channels}, received C={c}"
        )

    if expected_size is not None and (h != expected_size or w != expected_size):
        raise ValueError(
            f"Unexpected spatial size: expected {expected_size}x{expected_size}, "
            f"received {h}x{w}"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """
    CLI entry point.

    The function loads a split YAML, builds the real Dataset/DataLoader pipeline,
    iterates over batches, validates shapes and label values, and prints a
    compact diagnostic summary.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples",
        type=str,
        required=True,
        help="Split YAML path, e.g. configs/splits/spot_train_samples.yaml.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="crossearth",
        help="Model name used by pipeline.py.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Logical Dataset split. train=random crop, val/test=grid crop.",
    )

    parser.add_argument(
        "--patch-size-px",
        type=int,
        default=512,
        help="Crop size in pixels before model-aware resizing.",
    )

    parser.add_argument(
        "--patch-size-m",
        type=float,
        default=None,
        help="Crop size in meters. Use only when physical crop size is needed.",
    )

    parser.add_argument(
        "--stride-px",
        type=int,
        default=None,
        help="Stride for val/test. If None, patch-size-px is used.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="On Windows, start from 0.",
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=10,
        help="Maximum number of batches to check. 0 = all.",
    )

    parser.add_argument(
        "--max-invalid-frac",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--min-valid-frac",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--min-valid-classes",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--sensor-batch-sampler",
        action="store_true",
        help="Use mono-sensor batches.",
    )

    parser.add_argument(
        "--expected-channels",
        type=int,
        default=None,
        help="Expected number of channels. For CrossEarth RGBNIR use 4.",
    )

    parser.add_argument(
        "--expected-size",
        type=int,
        default=None,
        help="Expected spatial size. CrossEarth=504, DeepLab/SegFormer=512, DOFA=224.",
    )

    parser.add_argument(
        "--train-transform",
        action="store_true",
        help="Use training augmentation. By default EvalTransform is used.",
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load samples
    # ------------------------------------------------------------------

    samples = load_yaml(args.samples)

    if not isinstance(samples, list):
        raise ValueError(
            f"The file {args.samples} must contain a list of YAML samples."
        )

    if len(samples) == 0:
        raise ValueError(f"The split file is empty: {args.samples}")

    print("\n" + "=" * 80)
    print("CHECK DATASET")
    print("=" * 80)
    print("Samples file :", args.samples)
    print("Model name   :", args.model_name)
    print("Split        :", args.split)
    print("Patch px     :", args.patch_size_px)
    print("Patch m      :", args.patch_size_m)
    print("Stride px    :", args.stride_px)
    print("Batch size   :", args.batch_size)
    print("Num workers  :", args.num_workers)
    print("=" * 80)

    sample_summary = summarize_samples(samples)

    print("\nSamples summary:")
    print("  num_samples:", sample_summary["num_samples"])
    print("  by_sensor  :", sample_summary["by_sensor"])
    print("  first      :", sample_summary["first_sample"])

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    transform = (
        SegmentationTrainTransform()
        if args.train_transform
        else SegmentationEvalTransform()
    )

    dataset = MultiSensorSegDataset(
        samples=samples,
        model_name=args.model_name,
        split=args.split,
        patch_size_px=args.patch_size_px,
        patch_size_m=args.patch_size_m,
        stride_px=args.stride_px,
        max_invalid_frac=args.max_invalid_frac,
        min_valid_frac=args.min_valid_frac,
        min_valid_classes=args.min_valid_classes,
        max_retries=20,
        remap_labels=True,
        ignore_index=IGNORE_INDEX,
        transform=transform,
        return_meta=True,
    )

    print("\nDataset description:")
    print(dataset.describe())

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------

    if args.sensor_batch_sampler:
        batch_sampler = SensorBatchSampler(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=(args.split == "train"),
            drop_last=(args.split == "train"),
        )

        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=segmentation_collate,
        )

    else:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(args.split == "train"),
            drop_last=(args.split == "train"),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=segmentation_collate,
        )

    print("\nDataLoader:")
    print("  num_batches:", len(loader))
    print("  sampler    :", "SensorBatchSampler" if args.sensor_batch_sampler else "standard")

    # ------------------------------------------------------------------
    # Iterate
    # ------------------------------------------------------------------

    total_batches = 0
    total_samples = 0
    errors = []

    global_label_counts: Dict[int, int] = {}

    global_min = float("inf")
    global_max = float("-inf")
    global_sum = 0.0
    global_numel = 0

    print("\nChecking batches...\n")

    for batch_idx, batch in enumerate(loader):
        if 0 < args.max_batches <= batch_idx:
            break

        try:
            validate_batch(
                batch=batch,
                expected_channels=args.expected_channels,
                expected_size=args.expected_size,
            )

            images = batch["image"]
            labels = batch["label"]

            b = images.shape[0]

            batch_min = float(images.min().item())
            batch_max = float(images.max().item())
            batch_mean = float(images.mean().item())

            global_min = min(global_min, batch_min)
            global_max = max(global_max, batch_max)
            global_sum += float(images.sum().item())
            global_numel += int(images.numel())

            label_counts = count_labels_in_batch(labels)

            for k, v in label_counts.items():
                global_label_counts[k] = global_label_counts.get(k, 0) + v

            total_batches += 1
            total_samples += b

            print("-" * 80)
            print(f"Batch {batch_idx}")
            print("  image shape :", tuple(images.shape))
            print("  label shape :", tuple(labels.shape))
            print("  image range :", f"min={batch_min:.6f}, max={batch_max:.6f}, mean={batch_mean:.6f}")
            print("  sensors     :", batch.get("sensor"))
            print("  wavelengths :", batch.get("wavelengths"))

            print("  labels:")
            print_label_counts(label_counts)

            if "meta" in batch:
                print("  first meta  :", batch["meta"][0])

        except Exception as e:
            msg = str(e)
            errors.append((batch_idx, msg))

            print(f"\n[ERROR] Batch {batch_idx}: {msg}")

            if args.stop_on_error:
                raise

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Checked batches :", total_batches)
    print("Checked samples :", total_samples)
    print("Errors          :", len(errors))

    if global_numel > 0:
        print("Image min       :", global_min)
        print("Image max       :", global_max)
        print("Image mean      :", global_sum / global_numel)

    print("\nGlobal label distribution:")
    print_label_counts(global_label_counts)

    if errors:
        print("\nErrors:")
        for batch_idx, msg in errors[:20]:
            print(f"  Batch {batch_idx}: {msg}")

        if len(errors) > 20:
            print(f"  ... {len(errors) - 20} more errors")

    print("=" * 80)

    if errors:
        raise RuntimeError(
            f"Dataset check completed with {len(errors)} errors."
        )


if __name__ == "__main__":
    main()