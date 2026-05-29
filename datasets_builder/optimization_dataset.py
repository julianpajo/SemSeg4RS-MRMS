"""
pk_seg/dataset_builders/optimization/optimization_dataset.py
-----------------------
Preprocessing pipeline that converts raw sensor rasters into optimized
LitData streaming chunks for semantic segmentation training.

A single optimization pass produces patch-512 chunks with all spectral bands
and wavelengths. Model-specific transformations are applied at training time.

Stored format
-------------
patch_size : 512
bands      : all available bands from the sensor
wavelengths: always stored

Requirements
------------
- Sensor YAML configuration files in ``configs/sensors/<sensor_name>.yaml``
- Split text files in ``configs/splits/<dataset_name>/{train,val,test}.txt``
  Each line must contain: ``image_s3_uri label_s3_uri sensor_name``

Authentication
--------------
Credentials are read from environment variables:

    export AWS_ACCESS_KEY_ID="..."
    export AWS_SECRET_ACCESS_KEY="..."
    export AWS_ENDPOINT_URL="https://s3.gra.io.cloud.ovh.net/"
    export AWS_DEFAULT_REGION="auto"

Required dependencies:

    python -m pip install s3fs tqdm lightning tifffile torch
"""

from __future__ import annotations

import io
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import s3fs
import tifffile
from lightning.data import optimize
from tqdm import tqdm

# NOTE: Custom modules (configs.sensor_configs) and torch are deliberately
# removed from the global scope to prevent unpicklable objects (e.g., SemLocks)
# from breaking the multiprocessing context during the 'spawn' start method.


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\julia\OneDrive\Desktop\planetek\SemSeg4RS-MRMS")
SPLITS_ROOT = PROJECT_ROOT / "configs" / "splits"
SENSOR_CONFIG_ROOT = PROJECT_ROOT / "configs" / "sensors"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "optimized"
LOGS_ROOT = PROJECT_ROOT / "output" / "logs"

S3_ENDPOINT_URL = "https://s3.gra.io.cloud.ovh.net/"
S3_REGION = "gra"


S3_ENDPOINT_URL = "https://s3.gra.io.cloud.ovh.net/"
S3_REGION = "gra"


# ============================================================
# Optimization config
# ============================================================

PATCH_SIZE = 512
BANDS = "all"
WAVELENGTHS = True

IGNORE_INDEX = 255
NUM_WORKERS = 8  # Scaled to maximize parallel I/O-bound S3 downloads
CHUNK_BYTES = "64MB"
SPLITS = ["train", "val", "test"]
SKIP_IF_EXISTS = True


# ============================================================
# Logging utilities — WITH FORCED FLUSH TO AVOID EMPTY LOG FILES
# ============================================================

_logger: logging.Logger | None = None


def setup_logging() -> Path:
    """Initialize the logging system, writing to both stdout and a persistent log file.

    The log file is named with the current timestamp so each run produces an
    independent file that survives Studio restarts or spot preemptions.

    Returns:
        Path to the log file being written.
    """
    global _logger

    LOGS_ROOT.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_ROOT / f"optimization_{timestamp}.log"

    _logger = logging.getLogger("optimization")
    _logger.setLevel(logging.DEBUG)
    _logger.handlers.clear()

    # File handler — persists across session disconnects
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    # Console handler — mirrors output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter("%(message)s"))

    _logger.addHandler(fh)
    _logger.addHandler(ch)

    _logger.info(f"[INFO] Log file: {log_path}")
    for handler in _logger.handlers:
        handler.flush()

    return log_path


def _get_logger() -> logging.Logger:
    """Return the active logger, initializing it lazily if needed."""
    global _logger
    if _logger is None:
        setup_logging()
    return _logger


def log(message: str) -> None:
    """Print and persist an info message with instant disk flush."""
    _get_logger().info(f"[INFO] {message}")
    for handler in _get_logger().handlers:
        handler.flush()


def warn(message: str) -> None:
    """Print and persist a warning message with instant disk flush."""
    _get_logger().warning(f"[WARNING] {message}")
    for handler in _get_logger().handlers:
        handler.flush()


def error_log(message: str) -> None:
    """Print, track Traceback and persist a critical error with instant disk flush."""
    _get_logger().error(f"[CRITICAL ERROR] {message}", exc_info=True)
    for handler in _get_logger().handlers:
        handler.flush()


def section(title: str, char: str = "=") -> None:
    """Print and persist a formatted section header."""
    _get_logger().info("\n" + char * 80)
    _get_logger().info(title)
    _get_logger().info(char * 80)
    for handler in _get_logger().handlers:
        handler.flush()


def elapsed(start_time: float) -> str:
    """Calculate elapsed seconds from a performance counter timestamp."""
    return f"{time.perf_counter() - start_time:.2f}s"


# ============================================================
# S3 filesystem (Lazy & Worker-safe)
# ============================================================

_WORKER_FS: s3fs.S3FileSystem | None = None


def get_worker_fs() -> s3fs.S3FileSystem:
    """Initialize a lazy, worker-safe S3 filesystem instance."""
    global _WORKER_FS
    if _WORKER_FS is None:
        access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        endpoint_url = os.environ.get("AWS_ENDPOINT_URL", S3_ENDPOINT_URL)

        if not access_key or not secret_key:
            raise RuntimeError(
                "Missing S3 credentials in worker context. Ensure AWS_ACCESS_KEY_ID "
                "and AWS_SECRET_ACCESS_KEY environment variables are exported."
            )

        _WORKER_FS = s3fs.S3FileSystem(
            key=access_key,
            secret=secret_key,
            client_kwargs={
                "endpoint_url": endpoint_url,
                "region_name": "gra",
            },
            config_kwargs={
                "signature_version": "s3v4",
                "s3": {
                    "addressing_style": "path",
                }
            }
        )
    return _WORKER_FS


def strip_s3_protocol(path: str) -> str:
    """Remove the ``s3://`` protocol prefix from a URI string."""
    return path.removeprefix("s3://").rstrip("/")


# ============================================================
# Split reader
# ============================================================

def read_split_txt(txt_path: Path) -> List[Dict]:
    """Parse a dataset split file and build light-weight sample descriptors."""
    t0 = time.perf_counter()

    log(f"Reading split file: {txt_path}")

    if not txt_path.exists():
        raise FileNotFoundError(f"Split file not found: {txt_path}")

    file_size_mb = txt_path.stat().st_size / (1024 * 1024)
    log(f"Split file size: {file_size_mb:.2f} MB")

    with txt_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    log(f"Non-empty split rows found: {len(lines)}")

    samples: List[Dict] = []

    for line_idx, line in enumerate(
        tqdm(lines, desc=f"Parsing {txt_path.name}", unit="row"),
        start=0,
    ):
        parts = line.split()

        if len(parts) != 3:
            raise ValueError(
                f"Invalid split row at line {line_idx + 1} in {txt_path}: "
                f"expected 3 columns, got {len(parts)}. Row: {line}"
            )

        image_path, label_path, sensor_name = parts

        sample = {
            "image": image_path,
            "label": label_path,
            "sensor": sensor_name,
            "patch_size": PATCH_SIZE,
        }

        samples.append(sample)

    log(f"Finished reading split file in {elapsed(t0)}.")
    log(f"Samples built: {len(samples)}")

    if samples:
        first = samples[0]
        log("First sample preview:")
        log(f"   image : {first['image']}")
        log(f"   label : {first['label']}")
        log(f"   sensor: {first['sensor']}")

    return samples


# ============================================================
# I/O with Resilient S3 Retry Logic
# ============================================================

def _open_tiff_bytes_with_retry(path: str, retries: int = 5, delay: float = 1.5) -> np.ndarray:
    """Read a TIFF raster file from S3 or local storage with resilient retry backoff."""
    for attempt in range(retries):
        try:
            if path.startswith("s3://"):
                fs = get_worker_fs()
                with fs.open(strip_s3_protocol(path), "rb") as fh:
                    data = fh.read()
                return tifffile.imread(io.BytesIO(data))
            return tifffile.imread(path)
        except Exception as e:
            if attempt == retries - 1:
                raise e
            actual_delay = (delay * (attempt + 1)) + random.uniform(0.5, 2.0)
            time.sleep(actual_delay)


def read_image(path: str) -> np.ndarray:
    """Read a raster image and enforce a channel-first format ``(C, H, W)``."""
    image = _open_tiff_bytes_with_retry(path)

    if image.ndim == 2:
        image = image[..., None]

    if image.shape[-1] <= 32:
        image = np.moveaxis(image, -1, 0)

    return image


def read_label(path: str) -> np.ndarray:
    """Read a semantic mask file and enforce a 2-D map layout ``(H, W)``."""
    label = _open_tiff_bytes_with_retry(path)

    if label.ndim == 3:
        label = label[..., 0]

    return label


# ============================================================
# Sensor-aware preprocessing
# ============================================================

def normalize_image(image: np.ndarray, sensor_cfg: Dict) -> np.ndarray:
    """Normalize raw sensor imagery data into a clean float32 ``[0, 1]`` range."""
    image = image.astype(np.float32)

    scale_factor = sensor_cfg.get("scale_factor", None)
    nodata_value = sensor_cfg.get("nodata_value", None)

    if nodata_value is not None:
        nodata_mask = image == float(nodata_value)
    else:
        nodata_mask = np.zeros_like(image, dtype=bool)

    if scale_factor is not None:
        divisor = float(scale_factor)
    else:
        bit_depth = int(sensor_cfg.get("bit_depth", 16))
        divisor = float((2 ** bit_depth) - 1)

    if divisor <= 0:
        raise ValueError(f"Invalid normalization divisor calculated: {divisor}")

    image = image / divisor
    image[nodata_mask] = 0.0
    image = np.clip(image, 0.0, 1.0)
    np.nan_to_num(image, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return image.astype(np.float32)


def remap_label(label: np.ndarray, sensor_cfg: Dict) -> np.ndarray:
    """Convert raw sensor class integers to standard project targets."""
    mapping = sensor_cfg["label_mapping"]

    raw_invalid = int(mapping["invalid_pixel"])
    raw_sealed = int(mapping["sealed_soil"])
    raw_non_sealed = int(mapping["non_sealed_soil"])

    expected = {raw_invalid, raw_sealed, raw_non_sealed}
    found = set(np.unique(label).tolist())
    unexpected = found - expected

    if unexpected:
        raise ValueError(
            f"Unexpected raw labels found for sensor={sensor_cfg.get('name')}: "
            f"{unexpected}. Expected set is={expected}"
        )

    out = np.full(label.shape, IGNORE_INDEX, dtype=np.int64)
    out[label == raw_sealed] = 0
    out[label == raw_non_sealed] = 1

    return out


# ============================================================
# Spatial handling
# ============================================================

def center_crop_or_pad(
    image: np.ndarray,
    label: np.ndarray,
    target_size: int,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Ensure spatial geometry squares up perfectly with the target patch size."""
    _, h, w = image.shape

    if h < target_size or w < target_size:
        pad_h = max(0, target_size - h)
        pad_w = max(0, target_size - w)

        image = np.pad(
            image,
            pad_width=((0, 0), (0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=0,
        )

        label = np.pad(
            label,
            pad_width=((0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=IGNORE_INDEX,
        )

        _, h, w = image.shape

    y0 = max(0, (h - target_size) // 2)
    x0 = max(0, (w - target_size) // 2)

    image = image[:, y0:y0 + target_size, x0:x0 + target_size]
    label = label[y0:y0 + target_size, x0:x0 + target_size]

    return image, label, y0, x0


def is_valid_patch(label: np.ndarray) -> bool:
    """Verify if a candidate patch contains sufficient semantic labels."""
    valid_frac = float((label != IGNORE_INDEX).mean())

    if valid_frac < 0.10:
        return False

    semantic = label[label != IGNORE_INDEX]

    return semantic.size > 0


# ============================================================
# LitData optimize function
# ============================================================

def preprocess_tile(sample: dict) -> Iterator[dict]:
    """Execute isolated item preprocessing inside worker subprocesses."""
    # INLINE IMPORTS: Kept local to avoid cross-process initialization errors
    import torch
    from configs.sensor_configs import read_sensor_config_file, validate_sensor_config

    if not hasattr(preprocess_tile, "_sensor_cache"):
        preprocess_tile._sensor_cache = {}

    sensor_name = sample["sensor"]

    if sensor_name not in preprocess_tile._sensor_cache:
        try:
            sensor_config_path = SENSOR_CONFIG_ROOT / f"{sensor_name}.yaml"
            raw_cfg = read_sensor_config_file(sensor_config_path)
            sensor_cfg = validate_sensor_config(raw_cfg)
            preprocess_tile._sensor_cache[sensor_name] = sensor_cfg
        except Exception as exc:
            raise RuntimeError(f"Worker failed to load/validate YAML sensor configuration for {sensor_name}: {exc}")

    sensor_cfg = preprocess_tile._sensor_cache[sensor_name]

    try:
        image = read_image(sample["image"])
        label = read_label(sample["label"])
    except Exception as exc:
        print(f"[WARNING] Skipping sample due to permanent S3 I/O failure: {sample['image']}. Error details: {exc}", flush=True)
        return

    image = normalize_image(image, sensor_cfg)
    label = remap_label(label, sensor_cfg)

    image, label, y0, x0 = center_crop_or_pad(
        image=image,
        label=label,
        target_size=sample["patch_size"],
    )

    if not is_valid_patch(label):
        return

    wavelengths = sensor_cfg.get("wavelengths", None)
    if wavelengths is None:
        raise ValueError(f"Missing wavelengths metadata in sensor configuration: {sensor_name}.")

    rgbnir_idx = sensor_cfg.get("rgbnir_idx", None)
    if rgbnir_idx is None:
        raise ValueError(f"Missing rgbnir_idx metadata in sensor configuration: {sensor_name}.")

    yield {
        "image": torch.from_numpy(image.copy()).float(),
        "label": torch.from_numpy(label.copy()).long(),
        "wavelengths": torch.tensor(wavelengths, dtype=torch.float32),
        "rgbnir_idx": torch.tensor(rgbnir_idx, dtype=torch.long),
        "sensor": sensor_name,
        "image_path": sample["image"],
        "label_path": sample["label"],
        "y0": int(y0),
        "x0": int(x0),
        "patch_size": int(sample["patch_size"]),
    }


# ============================================================
# Discovery
# ============================================================

def discover_split_datasets() -> List[Path]:
    """Scan SPLITS_ROOT to locate dataset configurations containing train splits."""
    log(f"Discovering split datasets under: {SPLITS_ROOT}")

    if not SPLITS_ROOT.exists():
        raise FileNotFoundError(f"SPLITS_ROOT directory does not exist: {SPLITS_ROOT}")

    dataset_dirs = sorted(
        path for path in SPLITS_ROOT.iterdir()
        if path.is_dir() and (path / "train.txt").exists()
    )

    log(f"Discovered dataset split directories: {len(dataset_dirs)}")

    for path in dataset_dirs:
        log(f"  - {path}")

    return dataset_dirs


def optimized_exists(output_dir: Path) -> bool:
    """Evaluate whether an optimized folder contains a *complete* LitData dataset."""
    log(f"Checking whether optimized output already exists: {output_dir}")

    t0 = time.perf_counter()

    index_file = output_dir / "index.json"
    exists = index_file.exists()

    if exists:
        log(f"index.json found — dataset is complete: {index_file}")
    else:
        log(f"index.json missing — dataset is absent or incomplete: {output_dir}")

    log(f"Output existence check completed in {elapsed(t0)}.")
    return exists


# ============================================================
# Main execution loop — with comprehensive global error handling
# ============================================================

def main() -> None:
    """Run the batch LitData optimization pipeline across splits and datasets."""
    global_start = time.perf_counter()

    # Initialize logging to both file and stdout
    log_path = setup_logging()

    try:
        section("CREATING OPTIMIZED DATASETS")
        log(f"Log file       : {log_path}")
        log(f"Project root   : {PROJECT_ROOT}")
        log(f"Splits root    : {SPLITS_ROOT}")
        log(f"Output root    : {OUTPUT_ROOT}")
        log(f"Patch size     : {PATCH_SIZE}")
        log(f"Bands          : {BANDS}")
        log(f"Wavelengths    : {WAVELENGTHS}")
        log(f"Workers        : {NUM_WORKERS}")
        log(f"Chunk bytes    : {CHUNK_BYTES}")
        log(f"Skip if exists : {SKIP_IF_EXISTS}")
        log(f"S3 endpoint    : {os.environ.get('AWS_ENDPOINT_URL', S3_ENDPOINT_URL)}")

        print("=" * 80, flush=True)
        print("Training-time transforms per model:", flush=True)
        print("  DOFA          — no transform", flush=True)
        print("  CrossEarth    — center-crop 512 → 504", flush=True)
        print("  DeepLabV3+    — select rgbnir bands", flush=True)
        print("  SegFormer-SAE — select rgbnir bands", flush=True)
        print("=" * 80, flush=True)

        section("DISCOVERING SPLITS")
        dataset_dirs = discover_split_datasets()

        section("VERIFYING S3 ENV")
        if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
            raise RuntimeError(
                "Missing required S3 credentials in environment. Export AWS_ACCESS_KEY_ID "
                "and AWS_SECRET_ACCESS_KEY before execution."
            )
        log("S3 credentials verified in main process.")

        for dataset_dir in dataset_dirs:
            dataset_name = dataset_dir.name

            section(f"DATASET: {dataset_name}", char="#")

            for split in SPLITS:
                split_start = time.perf_counter()

                section(f"PROCESSING SPLIT: {dataset_name}/{split}", char="-")

                split_txt = dataset_dir / f"{split}.txt"
                output_dir = OUTPUT_ROOT / dataset_name / split

                log(f"Split file : {split_txt}")
                log(f"Output dir : {output_dir}")

                log("Checking split file existence...")
                if not split_txt.exists():
                    warn(f"Skipping split because split file is missing: {split_txt}")
                    continue
                log("Split file exists.")

                if SKIP_IF_EXISTS:
                    log("SKIP_IF_EXISTS is enabled. Checking output directory...")
                    if optimized_exists(output_dir):
                        warn(f"Skipping split because optimized output already exists: {output_dir}")
                        continue
                    log("No existing optimized output found. Continuing.")
                else:
                    log("SKIP_IF_EXISTS is disabled. Existing output will not be skipped.")

                log("Reading split file and building sample descriptors...")
                samples = read_split_txt(split_txt)

                if not samples:
                    warn(f"Skipping split because no samples were parsed from: {split_txt}")
                    continue

                log(f"Samples ready: {len(samples)}")
                log("Starting LitData optimize(). This step reads S3 TIFFs and writes chunks.")

                opt_start = time.perf_counter()
                real_output_dir = str(output_dir.resolve())
                log(f"Resolved local output directory: {real_output_dir}")

                # Enable regional Boto3 workaround for local paths
                old_region = os.environ.get("AWS_DEFAULT_REGION")
                os.environ["AWS_DEFAULT_REGION"] = "auto"

                try:
                    optimize(
                        fn=preprocess_tile,
                        inputs=samples,
                        output_dir=real_output_dir,
                        num_workers=NUM_WORKERS,
                        chunk_bytes=CHUNK_BYTES,
                    )
                except Exception as optimize_exc:
                    error_log(f"LitData optimize() raised a critical exception: {str(optimize_exc)}")
                    raise optimize_exc
                finally:
                    # Immediately restore environment variables to avoid side effects
                    if old_region:
                        os.environ["AWS_DEFAULT_REGION"] = old_region
                    else:
                        os.environ.pop("AWS_DEFAULT_REGION", None)

                log(f"LitData optimize() completed in {elapsed(opt_start)}.")

                # Final check that the index.json manifest file was written
                index_file = output_dir / "index.json"
                if not index_file.exists():
                    warn(
                        f"INCOMPLETE: index.json missing after optimize() — "
                        f"output may be corrupt and will be reprocessed next run: {output_dir}"
                    )
                else:
                    log(f"Verified complete: index.json present at {output_dir}")

                log(f"Split completed in {elapsed(split_start)}.")

    except KeyboardInterrupt:
        warn("The optimization pipeline was manually interrupted by the user (Ctrl+C).")
    except Exception as general_exc:
        error_log(f"Unexpected fatal error during pipeline execution: {str(general_exc)}")
        raise general_exc
    finally:
        section("OPTIMIZATION COMPLETE")
        log(f"Total pipeline runtime: {elapsed(global_start)}")


if __name__ == "__main__":
    main()