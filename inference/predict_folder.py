"""
inference/predict_folder.py
---------------------------

Inference on all GeoTIFF files inside a folder using one CrossEarth checkpoint.

Example
-------
    python inference/predict_folder.py ^
      --input-dir data/raw/planetscope/images ^
      --sensor-config configs/sensors/planetscope.yaml ^
      --checkpoint outputs/checkpoints/crossearth_finetune/best.pth ^
      --output-dir outputs/predictions/crossearth/planetscope ^
      --device cuda:0 ^
      --recursive ^
      --skip-existing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import rasterio
import torch
from rasterio.transform import Affine
from tqdm import tqdm

from configs.sensor_configs import read_sensor
from preprocessing.preprocess import preprocess_image_for_model
from models.crossearth import CrossEarthSeg


# ---------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------

def load_checkpoint_state(path: str | Path) -> Dict[str, torch.Tensor]:
    """
    Load the model state_dict from a checkpoint file.
    """
    ckpt = torch.load(path, map_location="cpu")

    if isinstance(ckpt, dict):
        for key in ["model", "model_state_dict", "state_dict"]:
            if key in ckpt:
                return ckpt[key]

    if isinstance(ckpt, dict):
        return ckpt

    raise RuntimeError(f"Unrecognized checkpoint format: {path}")


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remove the 'module.' prefix from checkpoint keys, if present.
    """
    out = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v

    return out


# ---------------------------------------------------------------------
# Raster utilities
# ---------------------------------------------------------------------

def update_transform_after_resize(
    old_transform,
    old_height: int,
    old_width: int,
    new_height: int,
    new_width: int,
):
    """
    Update affine transform after resizing while preserving geographic extent.
    """
    scale_x = old_width / new_width
    scale_y = old_height / new_height

    return old_transform * Affine.scale(scale_x, scale_y)


def make_invalid_mask(x_raw: np.ndarray, nodata_value) -> np.ndarray:
    """
    Create invalid mask from raster nodata value.
    """
    if nodata_value is None:
        return np.zeros(x_raw.shape[-2:], dtype=bool)

    return np.all(x_raw == nodata_value, axis=0)


def find_geotiffs(input_dir: Path, recursive: bool) -> List[Path]:
    """
    Find GeoTIFF files inside a folder.
    """
    patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]

    files: List[Path] = []

    for pattern in patterns:
        if recursive:
            files.extend(input_dir.rglob(pattern))
        else:
            files.extend(input_dir.glob(pattern))

    return sorted(set(files))


def build_output_path(
    image_path: Path,
    input_dir: Path,
    output_dir: Path,
    suffix: str,
    recursive: bool,
) -> Path:
    """
    Build output prediction path.

    If recursive=True, the input folder structure is preserved.
    """
    if recursive:
        rel = image_path.relative_to(input_dir)
        return output_dir / rel.parent / f"{rel.stem}{suffix}.tif"

    return output_dir / f"{image_path.stem}{suffix}.tif"


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def build_crossearth_model(device: torch.device) -> CrossEarthSeg:
    """
    Build CrossEarth model with RGBNIR configuration.
    """
    model = CrossEarthSeg.from_pretrained(
        variant="dinov2_vitl14_reg",
        num_classes=2,
        in_channels=4,
        decoder="mla",
        patch_embed_init="rgb_mean",
        freeze_backbone=True,
        train_patch_embed=True,
    )

    model = model.to(device)
    model.eval()

    return model


def load_model(checkpoint_path: Path, device: torch.device) -> CrossEarthSeg:
    """
    Build CrossEarth and load checkpoint weights.
    """
    model = build_crossearth_model(device)

    state = load_checkpoint_state(checkpoint_path)
    state = strip_module_prefix(state)

    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print(f"[WARNING] Missing keys: {len(missing)}")
        print("  first:", missing[:5])

    if unexpected:
        print(f"[WARNING] Unexpected keys: {len(unexpected)}")
        print("  first:", unexpected[:5])

    model.eval()

    return model


# ---------------------------------------------------------------------
# Single image prediction
# ---------------------------------------------------------------------

@torch.no_grad()
def predict_one(
    image_path: Path,
    output_path: Path,
    sensor_config_path: Path,
    model: CrossEarthSeg,
    device: torch.device,
    output_mode: str,
    apply_nodata_mask: bool,
    amp: bool,
) -> None:
    """
    Run inference on one GeoTIFF and save prediction.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(image_path) as src:
        x_raw = src.read()
        profile = src.profile.copy()
        old_height = src.height
        old_width = src.width
        old_transform = src.transform
        nodata_value = src.nodata

    invalid_mask_raw = make_invalid_mask(x_raw, nodata_value)

    info = read_sensor(
        image_path=str(image_path),
        sensor_config_path=str(sensor_config_path),
    )

    x_proc, _ = preprocess_image_for_model(
        x=x_raw,
        info=info,
        model_name="crossearth",
    )

    x_tensor = torch.from_numpy(x_proc).float().unsqueeze(0).to(device)

    with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and amp)):
        logits = model(x_tensor)

    pred_train = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

    if output_mode == "train":
        pred_out = pred_train
        nodata_out = 255

    elif output_mode == "qgis":
        pred_out = np.zeros_like(pred_train, dtype=np.uint8)
        pred_out[pred_train == 0] = 1
        pred_out[pred_train == 1] = 2
        nodata_out = 0

    else:
        raise ValueError(f"Invalid output_mode: {output_mode}")

    if apply_nodata_mask and invalid_mask_raw.any():
        mask = torch.from_numpy(invalid_mask_raw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        mask = torch.nn.functional.interpolate(
            mask,
            size=pred_out.shape,
            mode="nearest",
        )
        mask = mask.squeeze().numpy().astype(bool)

        if output_mode == "train":
            pred_out[mask] = 255
        else:
            pred_out[mask] = 0

    new_height, new_width = pred_out.shape

    new_transform = update_transform_after_resize(
        old_transform=old_transform,
        old_height=old_height,
        old_width=old_width,
        new_height=new_height,
        new_width=new_width,
    )

    profile.update(
        {
            "count": 1,
            "height": new_height,
            "width": new_width,
            "dtype": "uint8",
            "nodata": nodata_out,
            "transform": new_transform,
            "compress": "lzw",
        }
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(pred_out, 1)


# ---------------------------------------------------------------------
# Folder prediction
# ---------------------------------------------------------------------

def predict_folder(args: argparse.Namespace) -> None:
    """
    Run inference on all GeoTIFF files in a folder.
    """
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    sensor_config_path = Path(args.sensor_config)
    checkpoint_path = Path(args.checkpoint)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    if not sensor_config_path.exists():
        raise FileNotFoundError(f"Sensor config not found: {sensor_config_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_geotiffs(input_dir, recursive=args.recursive)

    if len(files) == 0:
        raise RuntimeError(f"No GeoTIFF files found in: {input_dir}")

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("=" * 80)
    print("CROSSEARTH FOLDER INFERENCE")
    print("=" * 80)
    print("Input dir    :", input_dir)
    print("Output dir   :", output_dir)
    print("Sensor config:", sensor_config_path)
    print("Checkpoint   :", checkpoint_path)
    print("Device       :", device)
    print("Files        :", len(files))
    print("Recursive    :", args.recursive)
    print("Output mode  :", args.output_mode)
    print("Skip existing:", args.skip_existing)
    print("=" * 80)

    model = load_model(checkpoint_path, device)

    errors = []

    for image_path in tqdm(files, desc="Predicting"):
        output_path = build_output_path(
            image_path=image_path,
            input_dir=input_dir,
            output_dir=output_dir,
            suffix=args.suffix,
            recursive=args.recursive,
        )

        if args.skip_existing and output_path.exists():
            continue

        try:
            predict_one(
                image_path=image_path,
                output_path=output_path,
                sensor_config_path=sensor_config_path,
                model=model,
                device=device,
                output_mode=args.output_mode,
                apply_nodata_mask=args.apply_nodata_mask,
                amp=args.amp,
            )

        except Exception as exc:
            errors.append(
                {
                    "image": str(image_path),
                    "output": str(output_path),
                    "error": str(exc),
                }
            )

            print(f"\n[ERROR] {image_path}: {exc}")

            if args.stop_on_error:
                raise

    print("=" * 80)
    print("DONE")
    print("Processed :", len(files) - len(errors))
    print("Errors    :", len(errors))
    print("Output dir:", output_dir)

    if errors:
        print("\nFirst errors:")
        for err in errors[:10]:
            print(f"  {err['image']} -> {err['error']}")

    print("=" * 80)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for folder inference.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Folder containing raw input GeoTIFF files.",
    )

    parser.add_argument(
        "--sensor-config",
        type=str,
        required=True,
        help="Sensor YAML file, e.g. configs/sensors/planetscope.yaml.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Model checkpoint, e.g. outputs/checkpoints/crossearth/best.pth.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Folder where prediction GeoTIFF files will be saved.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: cuda:0, cuda:1, or cpu.",
    )

    parser.add_argument(
        "--output-mode",
        type=str,
        default="qgis",
        choices=["qgis", "train"],
        help=(
            "qgis: save 0=invalid, 1=sealed, 2=non_sealed. "
            "train: save 0=sealed, 1=non_sealed, 255=ignore."
        ),
    )

    parser.add_argument(
        "--suffix",
        type=str,
        default="_pred",
        help="Suffix added to each output filename.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search GeoTIFF files recursively and preserve folder structure.",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files whose output prediction already exists.",
    )

    parser.add_argument(
        "--apply-nodata-mask",
        action="store_true",
        help="Apply raster nodata value as invalid mask in the output.",
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use mixed precision during inference.",
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when a file fails.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_folder(args)