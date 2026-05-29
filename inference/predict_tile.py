"""
inference/predict_tile.py
-------------------------

Inference on a single raw GeoTIFF.

Example
-------
    python inference/predict_tile.py ^
      --image data/raw/planetscope/images/S4-02-01a-VA1_train_autumn_gt-012-015_512_7680.tif ^
      --sensor-config configs/sensors/planetscope.yaml ^
      --checkpoint outputs/checkpoints/crossearth/best.pth ^
      --output outputs/predictions/crossearth_planetscope_pred.tif ^
      --device cuda:0

Output GeoTIFF
--------------
    0 = invalid / nodata, if applied
    1 = sealed_soil
    2 = non_sealed_soil

Notes
-----
    The model predicts internally:

        0 = sealed_soil
        1 = non_sealed_soil

    This script remaps labels for QGIS visualization:

        0 -> 1
        1 -> 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import rasterio
import torch
from rasterio.transform import Affine

from configs.sensor_configs import read_sensor
from datasets.preprocessing.pipeline import preprocess_image_for_model
from models.crossearth import CrossEarthSeg


# ---------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------

def load_checkpoint_state(path: str | Path) -> Dict[str, torch.Tensor]:
    """
    Load the model state_dict from a checkpoint file.

    Supported checkpoint formats are:

        checkpoint["model"]
        checkpoint["model_state_dict"]
        checkpoint["state_dict"]
        checkpoint directly as a state_dict

    Parameters
    ----------
    path :
        Path to the checkpoint file.

    Returns
    -------
    dict
        Model state_dict.

    Raises
    ------
    RuntimeError
        If the checkpoint format is not recognized.
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

    This is needed when a checkpoint was saved from a model wrapped with
    DataParallel or DistributedDataParallel.

    Parameters
    ----------
    state_dict :
        Input checkpoint state_dict.

    Returns
    -------
    dict
        State_dict with normalized key names.
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
    Update the raster affine transform after resizing.

    The updated transform preserves the same geographic extent as the
    original raster while adapting pixel size to the new raster dimensions.

    Parameters
    ----------
    old_transform :
        Original raster affine transform.
    old_height :
        Original raster height.
    old_width :
        Original raster width.
    new_height :
        New raster height.
    new_width :
        New raster width.

    Returns
    -------
    affine.Affine
        Updated affine transform.
    """
    scale_x = old_width / new_width
    scale_y = old_height / new_height

    return old_transform * Affine.scale(scale_x, scale_y)


def make_invalid_mask(x_raw: np.ndarray, nodata_value) -> np.ndarray:
    """
    Create an invalid-pixel mask from the raster nodata value.

    A pixel is considered invalid only if all bands are equal to the nodata
    value.

    Parameters
    ----------
    x_raw :
        Raw raster array with shape (C, H, W).
    nodata_value :
        Nodata value read from the raster metadata.

    Returns
    -------
    np.ndarray
        Boolean mask with shape (H, W), where True indicates invalid pixels.
    """
    if nodata_value is None:
        return np.zeros(x_raw.shape[-2:], dtype=bool)

    return np.all(x_raw == nodata_value, axis=0)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def build_crossearth_model(device: torch.device) -> CrossEarthSeg:
    """
    Build the CrossEarth model with the same RGBNIR configuration used in training.

    Parameters
    ----------
    device :
        Torch device on which the model will be allocated.

    Returns
    -------
    CrossEarthSeg
        Initialized CrossEarth segmentation model in evaluation mode.
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


# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------

@torch.no_grad()
def predict(args: argparse.Namespace) -> None:
    """
    Run inference on a single GeoTIFF tile and save the prediction as GeoTIFF.

    The function performs the following steps:

        1. Read the input raster.
        2. Load the sensor configuration.
        3. Preprocess the image for CrossEarth.
        4. Build the model and load the checkpoint.
        5. Run forward inference.
        6. Remap training labels to the selected output label format.
        7. Optionally apply the nodata mask.
        8. Save the output prediction as a georeferenced GeoTIFF.

    Parameters
    ----------
    args :
        Parsed command-line arguments.
    """
    image_path = Path(args.image)
    sensor_config_path = Path(args.sensor_config)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not sensor_config_path.exists():
        raise FileNotFoundError(f"Sensor config not found: {sensor_config_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("=" * 80)
    print("CROSSEARTH INFERENCE")
    print("=" * 80)
    print("Image        :", image_path)
    print("Sensor config:", sensor_config_path)
    print("Checkpoint   :", checkpoint_path)
    print("Output       :", output_path)
    print("Device       :", device)
    print("=" * 80)

    # -----------------------------------------------------------------
    # Read raster
    # -----------------------------------------------------------------

    with rasterio.open(image_path) as src:
        x_raw = src.read()  # (C, H, W)
        profile = src.profile.copy()
        old_height = src.height
        old_width = src.width
        old_transform = src.transform
        nodata_value = src.nodata

    print("Raw shape:", x_raw.shape)
    print("Raw nodata:", nodata_value)

    invalid_mask_raw = make_invalid_mask(x_raw, nodata_value)

    # -----------------------------------------------------------------
    # Sensor configuration + datasets_builder
    # -----------------------------------------------------------------

    info = read_sensor(
        image_path=str(image_path),
        sensor_config_path=str(sensor_config_path),
    )

    x_proc, wavelengths = preprocess_image_for_model(
        x=x_raw,
        info=info,
        model_name="crossearth",
    )

    print("Preprocessed shape:", x_proc.shape)
    print("Wavelengths:", wavelengths)

    # -----------------------------------------------------------------
    # Model + checkpoint
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------

    x_tensor = torch.from_numpy(x_proc).float().unsqueeze(0).to(device)

    with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.amp)):
        logits = model(x_tensor)

    pred_train = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

    print("Logits shape:", tuple(logits.shape))
    print("Pred train labels:", np.unique(pred_train))

    # -----------------------------------------------------------------
    # Map training labels to QGIS/raw-style labels
    # -----------------------------------------------------------------
    # Training labels:
    #   0 = sealed_soil
    #   1 = non_sealed_soil
    #
    # Output labels:
    #   0 = invalid/nodata
    #   1 = sealed_soil
    #   2 = non_sealed_soil

    if args.output_mode == "train":
        pred_out = pred_train
        nodata_out = 255

    elif args.output_mode == "qgis":
        pred_out = np.zeros_like(pred_train, dtype=np.uint8)
        pred_out[pred_train == 0] = 1
        pred_out[pred_train == 1] = 2
        nodata_out = 0

    else:
        raise ValueError(f"Invalid output_mode: {args.output_mode}")

    # Apply nodata mask only if requested and if invalid pixels exist.
    # If datasets_builder resized the image, for example from 512 to 504,
    # the mask is resized with nearest-neighbor interpolation using torch.
    if args.apply_nodata_mask and invalid_mask_raw.any():
        mask = torch.from_numpy(invalid_mask_raw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        mask = torch.nn.functional.interpolate(
            mask,
            size=pred_out.shape,
            mode="nearest",
        )
        mask = mask.squeeze().numpy().astype(bool)

        if args.output_mode == "train":
            pred_out[mask] = 255
        else:
            pred_out[mask] = 0

    print("Output labels:", np.unique(pred_out))

    # -----------------------------------------------------------------
    # Save GeoTIFF
    # -----------------------------------------------------------------

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

    print("Saved:", output_path)
    print("=" * 80)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for single-tile inference.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Raw input GeoTIFF.",
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
        "--output",
        type=str,
        required=True,
        help="Output prediction GeoTIFF.",
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
        "--apply-nodata-mask",
        action="store_true",
        help="Apply the raster nodata value as invalid mask in the output.",
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use mixed precision during inference.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args)