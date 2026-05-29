"""
dataset_builders/optimization/create_splits.py
----------------------------------------------

Create train/validation/test split files for the SPOT dataset stored on an
OVH S3-compatible bucket.

This script no longer scans a local DATASETS_ROOT. The dataset root is
hardcoded as an S3 path and is expected to contain:

    s3://iride-lot-3/temp_stuff/mauro_sylos/training_datasets/soil_sealing/spot3m_v3.2_res1.5m/
    ├── images/
    └── labels/

Output layout
-------------
configs/splits/
└── spot/
    ├── train.txt
    ├── val.txt
    └── test.txt

Each split row has the following format:

    image_path label_path sensor_name

Example row:

    s3://.../images/tile_001.tif s3://.../labels/tile_001.tif spot

Authentication
--------------
Credentials are intentionally NOT hardcoded. Configure them in the environment:

    export AWS_ACCESS_KEY_ID="..."
    export AWS_SECRET_ACCESS_KEY="..."
    export AWS_ENDPOINT_URL="https://s3.gra.io.cloud.ovh.net/"
    export AWS_DEFAULT_REGION="gra"

Required dependencies:

    python -m pip install s3fs tqdm
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import s3fs
from tqdm import tqdm


# ============================================================
# Constants
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\julia\OneDrive\Desktop\planetek\SemSeg4RS-MRMS")

OUT_DIR = PROJECT_ROOT / "configs" / "splits"

SENSOR_NAME = "spot"

S3_ENDPOINT_URL = "https://s3.gra.io.cloud.ovh.net/"
S3_REGION = "gra"

DATASET_ROOT = (
    "s3://iride-lot-3/temp_stuff/mauro_sylos/"
    "training_datasets/soil_sealing/spot3m_v3.2_res1.5m"
)

IMAGES_DIR = f"{DATASET_ROOT}/images"
LABELS_DIR = f"{DATASET_ROOT}/labels"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SEED        = 42

SUPPORTED_IMAGE_SUFFIXES = (".tif", ".tiff")


# ============================================================
# Logging utilities
# ============================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_step(message: str) -> None:
    print(f"[INFO] {message}")


def print_warning(message: str) -> None:
    print(f"[WARNING] {message}")


# ============================================================
# S3 utilities
# ============================================================

def build_s3_filesystem() -> s3fs.S3FileSystem:
    """
    Build an S3 filesystem client for the OVH S3-compatible endpoint.

    Credentials are read from environment variables.

    Raises
    ------
    RuntimeError
        If required credentials are missing from the environment.
    """
    print_step("Checking S3 credentials from environment variables...")

    access_key  = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key  = os.environ.get("AWS_SECRET_ACCESS_KEY")
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL", S3_ENDPOINT_URL)
    region_name  = os.environ.get("AWS_DEFAULT_REGION", S3_REGION)

    if not access_key or not secret_key:
        raise RuntimeError(
            "Missing S3 credentials. Please export AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY before running this script."
        )

    print_step(f"Using S3 endpoint: {endpoint_url}")
    print_step(f"Using S3 region  : {region_name}")
    print_step("Creating S3 filesystem client...")

    fs = s3fs.S3FileSystem(
        key=access_key,
        secret=secret_key,
        client_kwargs={
            "endpoint_url": endpoint_url,
            "region_name":  region_name,
        },
    )

    print_step("S3 filesystem client created successfully.")
    return fs


def strip_s3_protocol(path: str) -> str:
    return path.removeprefix("s3://").rstrip("/")


def ensure_s3_protocol(path: str) -> str:
    return path if path.startswith("s3://") else f"s3://{path}"


def s3_exists(fs: s3fs.S3FileSystem, path: str) -> bool:
    return fs.exists(strip_s3_protocol(path))


def find_tif_files_s3(fs: s3fs.S3FileSystem, folder: str) -> List[str]:
    """
    Return a sorted list of all TIFF URIs under an S3 folder.

    Uses a single fs.find() call — no per-file round trips.
    """
    print_step(f"Scanning S3 folder for TIFF files: {folder}")

    folder_no_protocol = strip_s3_protocol(folder)
    files = fs.find(folder_no_protocol)

    print_step(f"Objects found under folder: {len(files)}")

    tif_files = sorted(
        ensure_s3_protocol(p)
        for p in files
        if p.lower().endswith(SUPPORTED_IMAGE_SUFFIXES)
    )

    print_step(f"TIFF files found: {len(tif_files)}")
    return tif_files


def get_filename(path: str) -> str:
    return path.rstrip("/").split("/")[-1]


# ============================================================
# KEY OPTIMISATION — bulk label listing
# ============================================================

def list_label_filenames(fs: s3fs.S3FileSystem) -> set[str]:
    """
    Return the set of *filenames* (not full paths) present in LABELS_DIR.

    A single fs.find() call replaces 500 k+ individual s3_exists() checks,
    reducing wall-clock time from several hours to a few minutes.
    """
    print_step(f"Bulk-listing label files from: {LABELS_DIR}")

    folder_no_protocol = strip_s3_protocol(LABELS_DIR)
    all_objects = fs.find(folder_no_protocol)

    label_filenames = {
        p.rstrip("/").split("/")[-1]
        for p in all_objects
        if p.lower().endswith(SUPPORTED_IMAGE_SUFFIXES)
    }

    print_step(f"Label files found: {len(label_filenames)}")
    return label_filenames


# ============================================================
# Split utilities
# ============================================================

def check_ratios() -> None:
    total = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    print_step(
        f"Checking split ratios: "
        f"train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}, total={total}"
    )
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1. Got {total}")


def save_split_txt(samples: List[Dict[str, str]], path: Path) -> None:
    print_step(f"Writing {len(samples)} samples to: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in tqdm(samples, desc=f"Writing {path.name}", unit="sample"):
            f.write(
                f"{sample['image']} "
                f"{sample['label']} "
                f"{sample['sensor']}\n"
            )
    print_step(f"Split file written successfully: {path}")


# ============================================================
# Discovery
# ============================================================

def validate_dataset_layout(fs: s3fs.S3FileSystem) -> None:
    print_header("VALIDATING DATASET LAYOUT")
    print_step(f"Dataset root: {DATASET_ROOT}")
    print_step(f"Images dir  : {IMAGES_DIR}")
    print_step(f"Labels dir  : {LABELS_DIR}")

    for label, path in [
        ("Dataset root", DATASET_ROOT),
        ("Images folder", IMAGES_DIR),
        ("Labels folder", LABELS_DIR),
    ]:
        print_step(f"Checking {label}...")
        if not s3_exists(fs, path):
            raise FileNotFoundError(f"{label} not found: {path}")

    print_step("Dataset layout validation completed successfully.")


def build_samples_for_dataset(
    fs: s3fs.S3FileSystem,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Build sample descriptors by matching images against a pre-fetched label set.

    The label directory is listed once (single fs.find() call).
    Matching is then done entirely in memory — O(1) per image.

    Returns
    -------
    Tuple[List[Dict[str, str]], List[str]]
        Valid sample dicts and a list of image paths whose label is missing.
    """
    print_header("DISCOVERING IMAGE/LABEL PAIRS")

    # --- 1. list images (one round trip) ---
    image_files = find_tif_files_s3(fs, IMAGES_DIR)
    if not image_files:
        raise RuntimeError(f"No .tif/.tiff images found in: {IMAGES_DIR}")

    # --- 2. list labels (one round trip, replaces 500k+ s3_exists calls) ---
    label_filenames = list_label_filenames(fs)

    # --- 3. in-memory match ---
    print_step("Matching image files with label files (in memory)...")

    samples: List[Dict[str, str]] = []
    missing_labels: List[str] = []

    for image_path in tqdm(image_files, desc="Matching labels", unit="image"):
        filename   = get_filename(image_path)
        label_path = f"{LABELS_DIR}/{filename}"

        if filename not in label_filenames:
            missing_labels.append(label_path)
            continue

        samples.append(
            {
                "image":  image_path,
                "label":  label_path,
                "sensor": SENSOR_NAME,
            }
        )

    print_step(f"Valid image/label pairs: {len(samples)}")
    print_step(f"Missing labels        : {len(missing_labels)}")

    if missing_labels:
        print_warning("Some images do not have a matching label.")
        print_warning("First missing labels:")
        for path in missing_labels[:10]:
            print(f"  - {path}")

    return samples, missing_labels


# ============================================================
# Split logic
# ============================================================

def split_samples(
    samples: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    print_header("CREATING TRAIN/VAL/TEST SPLITS")
    check_ratios()

    print_step(f"Using random seed: {SEED}")
    print_step(f"Total valid samples before shuffling: {len(samples)}")

    rng      = random.Random(SEED)
    shuffled = samples.copy()
    rng.shuffle(shuffled)

    n       = len(shuffled)
    n_train = int(round(n * TRAIN_RATIO))
    n_val   = int(round(n * VAL_RATIO))

    if n_train + n_val > n:
        n_val = max(0, n - n_train)

    train_samples = shuffled[:n_train]
    val_samples   = shuffled[n_train : n_train + n_val]
    test_samples  = shuffled[n_train + n_val :]

    print_step(f"Train samples: {len(train_samples)}")
    print_step(f"Val samples  : {len(val_samples)}")
    print_step(f"Test samples : {len(test_samples)}")

    return train_samples, val_samples, test_samples


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Create train/validation/test split files for the hardcoded SPOT S3 dataset.

    Optimised flow
    --------------
    1. Connect to the OVH S3-compatible endpoint.
    2. Validate dataset layout (3 existence checks).
    3. List images once   → one fs.find() call.
    4. List labels once   → one fs.find() call.
    5. Match in memory    → O(1) per image, no extra S3 round trips.
    6. Shuffle and split deterministically.
    7. Write split files.
    """
    print_header("CREATING S3 SINGLE-SENSOR SPLITS")
    print_step("Starting split generation script...")
    print_step(f"Project root : {PROJECT_ROOT}")
    print_step(f"Output root  : {OUT_DIR}")
    print_step(f"Sensor name  : {SENSOR_NAME}")
    print_step(f"Dataset root : {DATASET_ROOT}")

    fs = build_s3_filesystem()
    validate_dataset_layout(fs)

    samples, missing_labels = build_samples_for_dataset(fs)

    if not samples:
        raise RuntimeError("No valid image/label pairs were found. Aborting.")

    train_samples, val_samples, test_samples = split_samples(samples)

    dataset_out_dir = OUT_DIR / SENSOR_NAME

    print_header("WRITING SPLIT FILES")
    save_split_txt(train_samples, dataset_out_dir / "train.txt")
    save_split_txt(val_samples,   dataset_out_dir / "val.txt")
    save_split_txt(test_samples,  dataset_out_dir / "test.txt")

    print_header("SUMMARY")
    print(f"S3 endpoint    : {os.environ.get('AWS_ENDPOINT_URL', S3_ENDPOINT_URL)}")
    print(f"Dataset root   : {DATASET_ROOT}")
    print(f"Images dir     : {IMAGES_DIR}")
    print(f"Labels dir     : {LABELS_DIR}")
    print(f"Output dir     : {dataset_out_dir}")
    print(f"Sensor name    : {SENSOR_NAME}")
    print(f"Ratios         : train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}")
    print(f"Seed           : {SEED}")
    print(f"Valid samples  : {len(samples)}")
    print(f"Missing labels : {len(missing_labels)}")
    print(f"Train          : {len(train_samples)}")
    print(f"Val            : {len(val_samples)}")
    print(f"Test           : {len(test_samples)}")

    if missing_labels:
        print_warning("Missing labels preview:")
        for path in missing_labels[:10]:
            print(f"  - {path}")

    print_header("S3 SINGLE-SENSOR SPLITS CREATED SUCCESSFULLY")


if __name__ == "__main__":
    main()