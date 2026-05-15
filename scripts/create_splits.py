"""
scripts/create_splits.py
------------------------

Create train/validation/test splits for multi-sensor semantic segmentation.

Output
------
    configs/splits/train_samples.yaml
    configs/splits/val_samples.yaml
    configs/splits/test_samples.yaml
    configs/splits/split_summary.yaml

Generated sample format
-----------------------
    - image: data/raw/spot/images/tile_001.tif
      label: data/raw/spot/labels/tile_001.tif
      sensor: spot
      sensor_config: configs/sensors/spot.yaml

Single-sensor usage
-------------------
    python scripts/create_splits.py ^
        --images data/raw/spot/images ^
        --labels data/raw/spot/labels ^
        --sensor spot ^
        --sensor-config configs/sensors/spot.yaml ^
        --out-dir configs/splits ^
        --train-ratio 0.7 ^
        --val-ratio 0.15 ^
        --test-ratio 0.15 ^
        --seed 42

Multi-sensor usage
------------------
    python scripts/create_splits.py ^
        --datasets-root data/raw ^
        --sensors spot planetscope ^
        --out-dir configs/splits ^
        --train-ratio 0.7 ^
        --val-ratio 0.15 ^
        --test-ratio 0.15 ^
        --seed 42

Assumption
----------
Image and label files have the same filename.

Example
-------
    data/raw/spot/images/tile_001.tif
    data/raw/spot/labels/tile_001.tif
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def find_tif_files(folder: Path) -> List[Path]:
    """
    Find .tif/.tiff files inside a folder.

    Parameters
    ----------
    folder :
        Folder to scan.

    Returns
    -------
    list of pathlib.Path
        Sorted list of GeoTIFF files.
    """
    files = sorted(
        list(folder.glob("*.tif")) +
        list(folder.glob("*.tiff"))
    )

    return files


def to_posix(path: Path) -> str:
    """
    Convert a Path to a string using standard forward slashes.

    This is useful for portable YAML files.

    Parameters
    ----------
    path :
        Input path.

    Returns
    -------
    str
        POSIX-style path string.
    """
    return path.as_posix()


def check_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """
    Check that split ratios are valid.

    Parameters
    ----------
    train_ratio :
        Fraction assigned to the training split.
    val_ratio :
        Fraction assigned to the validation split.
    test_ratio :
        Fraction assigned to the test split.

    Raises
    ------
    ValueError
        If any ratio is negative or if the sum is not equal to 1.
    """
    ratios = [train_ratio, val_ratio, test_ratio]

    if any(r < 0 for r in ratios):
        raise ValueError(
            f"Ratios must be >= 0. Received: {ratios}"
        )

    total = train_ratio + val_ratio + test_ratio

    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must be 1. "
            f"Received: {total}"
        )


def save_yaml(data, path: Path) -> None:
    """
    Save data to a YAML file, creating the parent folder first.

    Parameters
    ----------
    data :
        Data structure to serialize.
    path :
        Output YAML path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
        )


# ---------------------------------------------------------------------
# Sample discovery
# ---------------------------------------------------------------------

def build_samples_for_sensor(
    images_dir: Path,
    labels_dir: Path,
    sensor: str,
    sensor_config: Path,
    make_relative_to: Path | None = None,
) -> Tuple[List[Dict], List[str]]:
    """
    Build samples for a single sensor.

    Images and labels are matched using the same filename.

    Parameters
    ----------
    images_dir :
        Directory containing image GeoTIFF files.
    labels_dir :
        Directory containing label GeoTIFF files.
    sensor :
        Sensor name to store in each sample.
    sensor_config :
        Path to the sensor YAML configuration file.
    make_relative_to :
        Optional base directory used to convert output paths to relative paths.

    Returns
    -------
    tuple
        samples :
            List of valid sample dictionaries.
        missing_labels :
            List of missing label paths.
    """
    if not images_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {images_dir}")

    if not labels_dir.exists():
        raise FileNotFoundError(f"Label folder not found: {labels_dir}")

    if not sensor_config.exists():
        raise FileNotFoundError(f"Sensor YAML not found: {sensor_config}")

    image_files = find_tif_files(images_dir)

    if len(image_files) == 0:
        raise RuntimeError(f"No .tif/.tiff file found in: {images_dir}")

    samples: List[Dict] = []
    missing_labels: List[str] = []

    for image_path in image_files:
        label_path = labels_dir / image_path.name

        if not label_path.exists():
            missing_labels.append(str(label_path))
            continue

        if make_relative_to is not None:
            image_out = image_path.resolve().relative_to(make_relative_to.resolve())
            label_out = label_path.resolve().relative_to(make_relative_to.resolve())
            sensor_config_out = sensor_config.resolve().relative_to(make_relative_to.resolve())
        else:
            image_out = image_path
            label_out = label_path
            sensor_config_out = sensor_config

        samples.append(
            {
                "image": to_posix(Path(image_out)),
                "label": to_posix(Path(label_out)),
                "sensor": sensor,
                "sensor_config": to_posix(Path(sensor_config_out)),
            }
        )

    return samples, missing_labels


def build_samples_single_sensor(args) -> Tuple[List[Dict], Dict]:
    """
    Build samples in single-sensor mode.

    Required arguments:

        --images
        --labels
        --sensor
        --sensor-config

    Parameters
    ----------
    args :
        Parsed command-line arguments.

    Returns
    -------
    tuple
        samples :
            List of sample dictionaries.
        summary :
            Discovery summary.
    """
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    sensor_config = Path(args.sensor_config)

    make_relative_to = Path(args.relative_to) if args.relative_to else None

    samples, missing_labels = build_samples_for_sensor(
        images_dir=images_dir,
        labels_dir=labels_dir,
        sensor=args.sensor,
        sensor_config=sensor_config,
        make_relative_to=make_relative_to,
    )

    summary = {
        "mode": "single_sensor",
        "sensor": args.sensor,
        "images_dir": to_posix(images_dir),
        "labels_dir": to_posix(labels_dir),
        "sensor_config": to_posix(sensor_config),
        "num_samples": len(samples),
        "num_missing_labels": len(missing_labels),
        "missing_labels_preview": missing_labels[:20],
    }

    return samples, summary


def build_samples_multi_sensor(args) -> Tuple[List[Dict], Dict]:
    """
    Build samples in multi-sensor mode.

    Required arguments:

        --datasets-root data/raw
        --sensors spot planetscope

    Expected structure:

        data/raw/{sensor}/images
        data/raw/{sensor}/labels
        configs/sensors/{sensor}.yaml

    Parameters
    ----------
    args :
        Parsed command-line arguments.

    Returns
    -------
    tuple
        samples :
            List of sample dictionaries from all sensors.
        summary :
            Discovery summary.
    """
    dataset_root = Path(args.dataset_root)
    sensor_config_root = Path(args.sensor_config_root)
    make_relative_to = Path(args.relative_to) if args.relative_to else None

    all_samples: List[Dict] = []
    per_sensor_summary = {}

    for sensor in args.sensors:
        images_dir = dataset_root / sensor / "images"
        labels_dir = dataset_root / sensor / "labels"
        sensor_config = sensor_config_root / f"{sensor}.yaml"

        samples, missing_labels = build_samples_for_sensor(
            images_dir=images_dir,
            labels_dir=labels_dir,
            sensor=sensor,
            sensor_config=sensor_config,
            make_relative_to=make_relative_to,
        )

        all_samples.extend(samples)

        per_sensor_summary[sensor] = {
            "images_dir": to_posix(images_dir),
            "labels_dir": to_posix(labels_dir),
            "sensor_config": to_posix(sensor_config),
            "num_samples": len(samples),
            "num_missing_labels": len(missing_labels),
            "missing_labels_preview": missing_labels[:20],
        }

    summary = {
        "mode": "multi_sensor",
        "dataset_root": to_posix(dataset_root),
        "sensor_config_root": to_posix(sensor_config_root),
        "sensors": args.sensors,
        "num_samples": len(all_samples),
        "per_sensor": per_sensor_summary,
    }

    return all_samples, summary


# ---------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------

def split_samples(
    samples: List[Dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    stratify_by_sensor: bool = True,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split samples into train/validation/test sets.

    If stratify_by_sensor=True, splitting is performed separately for each
    sensor to keep similar proportions across sensors.

    If False, all samples are shuffled and split together.

    Parameters
    ----------
    samples :
        List of sample dictionaries.
    train_ratio :
        Fraction assigned to the training split.
    val_ratio :
        Fraction assigned to the validation split.
    test_ratio :
        Fraction assigned to the test split.
    seed :
        Random seed used for deterministic shuffling.
    stratify_by_sensor :
        If True, preserve sensor proportions across splits.

    Returns
    -------
    tuple
        train_samples, val_samples, test_samples.
    """
    check_ratios(train_ratio, val_ratio, test_ratio)

    rng = random.Random(seed)

    train_samples: List[Dict] = []
    val_samples: List[Dict] = []
    test_samples: List[Dict] = []

    if stratify_by_sensor:
        groups: Dict[str, List[Dict]] = {}

        for sample in samples:
            sensor = sample.get("sensor", "unknown")
            groups.setdefault(sensor, []).append(sample)

        for sensor, group in groups.items():
            group = group.copy()
            rng.shuffle(group)

            n = len(group)
            n_train = int(round(n * train_ratio))
            n_val = int(round(n * val_ratio))

            # Avoid exceeding n because of rounding.
            if n_train + n_val > n:
                n_val = max(0, n - n_train)

            train_part = group[:n_train]
            val_part = group[n_train:n_train + n_val]
            test_part = group[n_train + n_val:]

            train_samples.extend(train_part)
            val_samples.extend(val_part)
            test_samples.extend(test_part)

    else:
        shuffled = samples.copy()
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        if n_train + n_val > n:
            n_val = max(0, n - n_train)

        train_samples = shuffled[:n_train]
        val_samples = shuffled[n_train:n_train + n_val]
        test_samples = shuffled[n_train + n_val:]

    # Final shuffle to avoid ordered sensor blocks.
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    return train_samples, val_samples, test_samples


def count_by_sensor(samples: List[Dict]) -> Dict[str, int]:
    """
    Count samples by sensor.

    Parameters
    ----------
    samples :
        List of sample dictionaries.

    Returns
    -------
    dict
        Mapping from sensor name to sample count.
    """
    counts: Dict[str, int] = {}

    for sample in samples:
        sensor = sample.get("sensor", "unknown")
        counts[sensor] = counts.get(sensor, 0) + 1

    return counts


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """
    CLI entry point.

    The function discovers image/label pairs, creates train/validation/test
    splits, saves the split YAML files and writes a summary YAML file.
    """
    parser = argparse.ArgumentParser()

    # Single-sensor mode.
    parser.add_argument(
        "--images",
        type=str,
        default=None,
        help="Image folder for single-sensor mode.",
    )

    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Label folder for single-sensor mode.",
    )

    parser.add_argument(
        "--sensor",
        type=str,
        default=None,
        help="Sensor name for single-sensor mode, e.g. spot.",
    )

    parser.add_argument(
        "--sensor-config",
        type=str,
        default=None,
        help="Sensor YAML file for single-sensor mode.",
    )

    # Multi-sensor mode.
    parser.add_argument(
        "--datasets-root",
        type=str,
        default=None,
        help="Multi-sensor datasets root, e.g. data/raw.",
    )

    parser.add_argument(
        "--sensors",
        nargs="+",
        default=None,
        help="Sensor list, e.g. spot planetscope.",
    )

    parser.add_argument(
        "--sensor-config-root",
        type=str,
        default="configs/sensors",
        help="Sensor YAML folder for multi-sensor mode.",
    )

    # Output.
    parser.add_argument(
        "--out-dir",
        type=str,
        default="configs/splits",
        help="Output split folder.",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional filename prefix. Example: spot_",
    )

    # Split ratios.
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--no-stratify-by-sensor",
        action="store_true",
        help="Disable sensor-stratified splitting.",
    )

    parser.add_argument(
        "--relative-to",
        type=str,
        default=".",
        help=(
            "Make paths relative to this folder. "
            "Default='.', meaning project root."
        ),
    )

    args = parser.parse_args()

    single_mode = (
        args.images is not None
        or args.labels is not None
        or args.sensor is not None
        or args.sensor_config is not None
    )

    multi_mode = (
        args.dataset_root is not None
        or args.sensors is not None
    )

    if single_mode and multi_mode:
        raise ValueError(
            "Use either single-sensor mode or multi-sensor mode, not both."
        )

    if single_mode:
        missing = []
        for name in ["images", "labels", "sensor", "sensor_config"]:
            if getattr(args, name) is None:
                missing.append(f"--{name.replace('_', '-')}")

        if missing:
            raise ValueError(
                f"Incomplete single-sensor mode. Missing: {missing}"
            )

        samples, discovery_summary = build_samples_single_sensor(args)

    elif multi_mode:
        if args.dataset_root is None or args.sensors is None:
            raise ValueError(
                "Multi-sensor mode requires --datasets-root and --sensors."
            )

        samples, discovery_summary = build_samples_multi_sensor(args)

    else:
        raise ValueError(
            "You must specify one mode:\n"
            "  Single sensor: --images --labels --sensor --sensor-config\n"
            "  Multi-sensor : --datasets-root --sensors"
        )

    if len(samples) == 0:
        raise RuntimeError("No valid sample found.")

    train_samples, val_samples, test_samples = split_samples(
        samples=samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify_by_sensor=not args.no_stratify_by_sensor,
    )

    out_dir = Path(args.out_dir)
    prefix = args.prefix

    train_path = out_dir / f"{prefix}train_samples.yaml"
    val_path = out_dir / f"{prefix}val_samples.yaml"
    test_path = out_dir / f"{prefix}test_samples.yaml"
    summary_path = out_dir / f"{prefix}split_summary.yaml"

    save_yaml(train_samples, train_path)
    save_yaml(val_samples, val_path)
    save_yaml(test_samples, test_path)

    summary = {
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "stratify_by_sensor": not args.no_stratify_by_sensor,
        "total_samples": len(samples),
        "splits": {
            "train": {
                "path": to_posix(train_path),
                "num_samples": len(train_samples),
                "by_sensor": count_by_sensor(train_samples),
            },
            "val": {
                "path": to_posix(val_path),
                "num_samples": len(val_samples),
                "by_sensor": count_by_sensor(val_samples),
            },
            "test": {
                "path": to_posix(test_path),
                "num_samples": len(test_samples),
                "by_sensor": count_by_sensor(test_samples),
            },
        },
        "discovery": discovery_summary,
    }

    save_yaml(summary, summary_path)

    print("\n" + "=" * 80)
    print("SPLIT CREATED")
    print("=" * 80)
    print(f"Total samples : {len(samples)}")
    print(f"Train         : {len(train_samples)} -> {train_path}")
    print(f"Val           : {len(val_samples)} -> {val_path}")
    print(f"Test          : {len(test_samples)} -> {test_path}")
    print(f"Summary       : {summary_path}")

    print("\nDistribution by sensor:")
    print("Train:", count_by_sensor(train_samples))
    print("Val  :", count_by_sensor(val_samples))
    print("Test :", count_by_sensor(test_samples))
    print("=" * 80)


if __name__ == "__main__":
    main()