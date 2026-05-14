"""
data/preprocess.py
------------------

Model-aware preprocessing transformations for multi-sensor remote-sensing
semantic segmentation.

Preprocessing depends on the target model:

  "all_bands" -> use all raster bands          [dofa]
  "rgbnir"    -> select R, G, B, NIR           [segformer_sae 4-band, DeepLabV3+]
  "rgb"       -> select R, G, B                [crossearth / DINO-like RGB]

Original dataset labels:

  0 = invalid_pixel
  1 = sealed_soil
  2 = non_sealed_soil

Training labels:

  255 = ignore_index
  0   = sealed_soil
  1   = non_sealed_soil

Therefore, the model must use:

  num_classes = 2

and the loss must be:

  torch.nn.CrossEntropyLoss(ignore_index=255)

Image normalization:

  - if SensorConfig contains scale_factor:
        x = x / scale_factor

    Typical example:
        PlanetScope Surface Reflectance -> scale_factor = 10000

  - otherwise:
        x = x / max_val

Resize:

  - bilinear interpolation for images
  - nearest-neighbor interpolation for labels
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from configs.sensor_configs import SensorConfig


# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------

TRAIN_SEALED_SOIL_LABEL = 0
TRAIN_NON_SEALED_SOIL_LABEL = 1

IGNORE_INDEX = 255


# ---------------------------------------------------------------------------
# Image normalization
# ---------------------------------------------------------------------------

def normalize(x: np.ndarray, info: SensorConfig) -> np.ndarray:
    """
    Normalize a raster array to float32.

    Parameters
    ----------
    x :
        Raster array with shape (C, H, W).
    info :
        Sensor configuration associated with the raster.

    Returns
    -------
    np.ndarray
        Normalized float32 array.

    Notes
    -----
    Strategy:

      - if info.scale_factor exists and is not None, use it;
      - otherwise, use info.max_val.

    For remote-sensing products, scale_factor is often more correct than the
    theoretical maximum value defined by the bit depth.

    Example:

        uint16 reflectance values scaled by 10000
        -> scale_factor = 10000
        -> not max_val = 65535
    """
    x = x.astype(np.float32)

    scale_factor = getattr(info, "scale_factor", None)
    nodata_value = getattr(info, "nodata_value", None)

    if nodata_value is not None:
        nodata_mask = x == nodata_value
    else:
        nodata_mask = np.zeros_like(x, dtype=bool)

    if scale_factor is not None:
        divisor = float(scale_factor)
    else:
        divisor = float(info.max_val)

    if divisor <= 0:
        raise ValueError(f"Invalid normalization divisor: {divisor}")

    x = x / divisor

    # Nodata pixels are set to 0 in the image.
    # The label must still mark them as ignore_index.
    x[nodata_mask] = 0.0

    x = np.clip(x, 0.0, 1.0)

    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# Band selection
# ---------------------------------------------------------------------------

def select_bands(x: np.ndarray, indices: List[int]) -> np.ndarray:
    """
    Select bands from an array with shape (C, H, W).

    Parameters
    ----------
    x :
        Image array with shape (C, H, W).
    indices :
        Band indices to select.

    Returns
    -------
    np.ndarray
        Array with shape (len(indices), H, W).
    """
    return x[indices]


def get_wavelengths_for(info: SensorConfig, indices: List[int]) -> List[float]:
    """
    Return wavelengths corresponding to the selected band indices.

    Parameters
    ----------
    info :
        Sensor configuration.
    indices :
        Selected band indices.

    Returns
    -------
    list of float
        Wavelengths associated with the selected bands.
    """
    return [info.wavelengths[i] for i in indices]


def validate_sensor_info(info: SensorConfig) -> None:
    """
    Run minimum consistency checks on the sensor configuration.

    Required checks:

      - n_bands must be consistent with wavelengths;
      - rgbnir_idx, if present, must contain 4 indices within the valid band range.

    Parameters
    ----------
    info :
        Sensor configuration to validate.
    """
    if len(info.wavelengths) != info.n_bands:
        raise ValueError(
            f"Inconsistent sensor configuration for {info.sensor_name}: "
            f"n_bands={info.n_bands}, but len(wavelengths)={len(info.wavelengths)}."
        )

    if info.rgbnir_idx is not None:
        if len(info.rgbnir_idx) != 4:
            raise ValueError(
                f"rgbnir_idx for {info.sensor_name} must contain 4 indices "
                f"in [R, G, B, NIR] order, received: {info.rgbnir_idx}"
            )

        if max(info.rgbnir_idx) >= info.n_bands or min(info.rgbnir_idx) < 0:
            raise ValueError(
                f"rgbnir_idx for {info.sensor_name} contains out-of-range indices: "
                f"{info.rgbnir_idx}, n_bands={info.n_bands}"
            )


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------

def resize_image(x: np.ndarray, target: int) -> np.ndarray:
    """
    Resize an image from (C, H, W) to (C, target, target).

    Bilinear interpolation is used.

    Parameters
    ----------
    x :
        Image array with shape (C, H, W).
    target :
        Target spatial size.

    Returns
    -------
    np.ndarray
        Resized image array.
    """
    if x.ndim != 3:
        raise ValueError(f"x must have shape (C, H, W), received shape={x.shape}")

    if x.shape[-2] == target and x.shape[-1] == target:
        return x

    t = torch.from_numpy(x).unsqueeze(0).float()

    t = F.interpolate(
        t,
        size=(target, target),
        mode="bilinear",
        align_corners=False,
    )

    return t.squeeze(0).numpy()


def resize_label(y: np.ndarray, target: int) -> np.ndarray:
    """
    Resize a label from (H, W) to (target, target).

    Nearest-neighbor interpolation is mandatory for semantic segmentation.

    Parameters
    ----------
    y :
        Label array with shape (H, W).
    target :
        Target spatial size.

    Returns
    -------
    np.ndarray
        Resized int64 label array.
    """
    if y.ndim != 2:
        raise ValueError(f"y must have shape (H, W), received shape={y.shape}")

    if y.shape[0] == target and y.shape[1] == target:
        return y.astype(np.int64)

    t = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    t = F.interpolate(
        t,
        size=(target, target),
        mode="nearest",
    )

    return t.squeeze(0).squeeze(0).numpy().astype(np.int64)


# ---------------------------------------------------------------------------
# Model requirements
# ---------------------------------------------------------------------------

MODEL_REQUIREMENTS = {
    "segformer_sae": {
        "band_mode": "rgbnir",
        "target_size": 512,
    },
    "deeplabv3plus": {
        "band_mode": "rgbnir",
        "target_size": 512,
    },
    "dofa": {
        "band_mode": "all_bands",
        "target_size": 224,
    },
    "crossearth": {
        "band_mode": "rgbnir",
        "target_size": 504,  # 504 = 36 x 14
    },
}


# ---------------------------------------------------------------------------
# Model-aware image pipeline
# ---------------------------------------------------------------------------

def preprocess_image(
    x: np.ndarray,
    info: SensorConfig,
    band_mode: str,
    target_size: int,
) -> Tuple[np.ndarray, List[float]]:
    """
    Fully preprocess an image for a specific model.

    Parameters
    ----------
    x :
        Raw array read from the raster, with shape (C, H, W).
    info :
        SensorConfig associated with the raster.
    band_mode :
        Band selection mode:

            "all_bands" -> all bands
            "rgbnir"    -> R, G, B, NIR
            "rgb"       -> R, G, B

    target_size :
        Final spatial size:

            target_size x target_size

    Returns
    -------
    tuple
        image :
            Normalized float32 array with shape (C_out, target_size, target_size).
        wavelengths :
            Wavelengths of the selected bands.
    """
    if x.ndim != 3:
        raise ValueError(
            f"x must have shape (C, H, W), received shape={x.shape}"
        )

    validate_sensor_info(info)

    if x.shape[0] != info.n_bands:
        raise ValueError(
            f"Inconsistent band count: image has {x.shape[0]} bands, "
            f"sensor config has {info.n_bands} bands."
        )

    # 1. Radiometric normalization
    x = normalize(x, info)

    # 2. Band selection
    if band_mode == "all_bands":
        indices = list(range(info.n_bands))
        wavelengths = list(info.wavelengths)

    elif band_mode == "rgbnir":
        if info.rgbnir_idx is None:
            raise ValueError(
                f"Sensor {info.sensor_name} does not define rgbnir_idx."
            )

        indices = list(info.rgbnir_idx)
        wavelengths = get_wavelengths_for(info, indices)

    elif band_mode == "rgb":
        if info.rgbnir_idx is None:
            raise ValueError(
                f"Sensor {info.sensor_name} does not define rgbnir_idx, "
                f"therefore RGB cannot be derived."
            )

        # rgbnir_idx is assumed to be in [R, G, B, NIR] order.
        indices = list(info.rgbnir_idx[:3])
        wavelengths = get_wavelengths_for(info, indices)

    else:
        raise ValueError(
            f"Invalid band_mode '{band_mode}'. "
            f"Choose one of: all_bands | rgbnir | rgb"
        )

    x = select_bands(x, indices)

    # 3. Spatial resize
    x = resize_image(x, target_size)

    return x.astype(np.float32), wavelengths


def preprocess_image_for_model(
    x: np.ndarray,
    info: SensorConfig,
    model_name: str,
) -> Tuple[np.ndarray, List[float]]:
    """
    Convenience wrapper that selects band_mode and target_size from model_name.

    Parameters
    ----------
    x :
        Raw image array with shape (C, H, W).
    info :
        Sensor configuration.
    model_name :
        Model name available in MODEL_REQUIREMENTS.

    Returns
    -------
    tuple
        Preprocessed image and selected wavelengths.
    """
    key = model_name.lower()

    if key not in MODEL_REQUIREMENTS:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Available: {list(MODEL_REQUIREMENTS.keys())}"
        )

    req = MODEL_REQUIREMENTS[key]

    return preprocess_image(
        x=x,
        info=info,
        band_mode=req["band_mode"],
        target_size=req["target_size"],
    )


# ---------------------------------------------------------------------------
# Label remapping
# ---------------------------------------------------------------------------

def remap_label_for_training(
    y: np.ndarray,
    info: SensorConfig,
    ignore_index: int = IGNORE_INDEX,
    check_unexpected: bool = True,
) -> np.ndarray:
    """
    Remap original sensor labels to the format used by the loss.

    The raw mapping is read from SensorConfig/YAML.

    Example SPOT:

        raw 0 = invalid_pixel
        raw 1 = sealed_soil
        raw 3 = non_sealed_soil

    Example PlanetScope:

        raw 0 = invalid_pixel
        raw 1 = sealed_soil
        raw 2 = non_sealed_soil

    Training labels:

        255 = ignore_index
        0   = sealed_soil
        1   = non_sealed_soil

    Parameters
    ----------
    y :
        Raw label array with shape (H, W).
    info :
        Sensor configuration containing label_mapping.
    ignore_index :
        Label value ignored by the loss.
    check_unexpected :
        If True, raise an error when labels not defined in label_mapping are found.

    Returns
    -------
    np.ndarray
        Remapped int64 label array.
    """
    if y.ndim != 2:
        raise ValueError(f"y must have shape (H, W), received shape={y.shape}")

    label_mapping = getattr(info, "label_mapping", None)

    if label_mapping is None:
        raise ValueError(
            f"SensorConfig for {info.sensor_name} does not contain label_mapping. "
            "Add label_mapping to the sensor YAML file."
        )

    required_keys = {"invalid_pixel", "sealed_soil", "non_sealed_soil"}
    missing_keys = required_keys - set(label_mapping.keys())

    if missing_keys:
        raise ValueError(
            f"Incomplete label_mapping for {info.sensor_name}. "
            f"Missing: {missing_keys}. "
            f"Required: {required_keys}"
        )

    raw_invalid = int(label_mapping["invalid_pixel"])
    raw_sealed = int(label_mapping["sealed_soil"])
    raw_non_sealed = int(label_mapping["non_sealed_soil"])

    expected_raw_labels = {raw_invalid, raw_sealed, raw_non_sealed}

    if check_unexpected:
        found_labels = set(np.unique(y).tolist())
        unexpected = found_labels - expected_raw_labels

        if unexpected:
            raise ValueError(
                f"Found unexpected raw labels for {info.sensor_name}: {unexpected}. "
                f"Expected: {expected_raw_labels}. "
                f"Check the sensor YAML file."
            )

    out = np.full_like(y, fill_value=ignore_index, dtype=np.int64)

    out[y == raw_sealed] = TRAIN_SEALED_SOIL_LABEL
    out[y == raw_non_sealed] = TRAIN_NON_SEALED_SOIL_LABEL

    # raw_invalid remains ignore_index

    return out.astype(np.int64)


def preprocess_label(
    y: np.ndarray,
    info: SensorConfig,
    target_size: int,
    remap: bool = True,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """
    Fully preprocess a label array.

    Steps
    -----
    1. nearest-neighbor resize;
    2. optional remapping for training.

    Parameters
    ----------
    y :
        Original label with shape (H, W), with values:

            0 = invalid_pixel
            1 = sealed_soil
            2 = non_sealed_soil

    info :
        Sensor configuration containing label_mapping.
    target_size :
        Final spatial size.
    remap :
        If True, remap labels for training:

            0 -> 255
            1 -> 0
            2 -> 1

        If False, preserve original labels.
    ignore_index :
        Value used for pixels ignored by the loss.

    Returns
    -------
    np.ndarray
        int64 label with shape (target_size, target_size).
    """
    if y.ndim != 2:
        raise ValueError(f"y must have shape (H, W), received shape={y.shape}")

    y = resize_label(y, target_size)

    if remap:
        y = remap_label_for_training(
            y=y,
            info=info,
            ignore_index=ignore_index,
        )
    else:
        y = y.astype(np.int64)

    return y


def preprocess_label_for_model(
    y: np.ndarray,
    info: SensorConfig,
    model_name: str,
    remap: bool = True,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """
    Convenience wrapper that preprocesses a label using the model target_size.

    Parameters
    ----------
    y :
        Raw label array with shape (H, W).
    info :
        Sensor configuration.
    model_name :
        Model name available in MODEL_REQUIREMENTS.
    remap :
        If True, remap labels for training.
    ignore_index :
        Value used for ignored pixels/classes.

    Returns
    -------
    np.ndarray
        Preprocessed label array.
    """
    key = model_name.lower()

    if key not in MODEL_REQUIREMENTS:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Available: {list(MODEL_REQUIREMENTS.keys())}"
        )

    target_size = MODEL_REQUIREMENTS[key]["target_size"]

    return preprocess_label(
        y=y,
        info=info,
        target_size=target_size,
        remap=remap,
        ignore_index=ignore_index,
    )


# ---------------------------------------------------------------------------
# Patch validity
# ---------------------------------------------------------------------------

def is_valid_patch(
    label: np.ndarray,
    info: SensorConfig,
    max_invalid_frac: float = 0.9,
    min_valid_frac: float = 0.1,
    min_valid_classes: int = 1,
) -> bool:
    """
    Check whether a raw label patch is usable for training.

    The function uses label_mapping defined in the sensor YAML file.

    Example SPOT:

        0 = invalid_pixel
        1 = sealed_soil
        3 = non_sealed_soil

    Example PlanetScope:

        0 = invalid_pixel
        1 = sealed_soil
        2 = non_sealed_soil

    A patch is discarded if:

      - it contains too many invalid pixels;
      - it contains too few valid pixels;
      - it contains fewer than min_valid_classes valid semantic classes.

    Parameters
    ----------
    label :
        Raw label patch with shape (H, W).
    info :
        Sensor configuration containing label_mapping.
    max_invalid_frac :
        Maximum allowed invalid-pixel fraction.
    min_valid_frac :
        Minimum required valid-pixel fraction.
    min_valid_classes :
        Minimum number of valid semantic classes required in the patch.

    Returns
    -------
    bool
        True if the patch is valid, False otherwise.
    """
    if label.ndim != 2:
        raise ValueError(
            f"label must have shape (H, W), received shape={label.shape}"
        )

    label_mapping = getattr(info, "label_mapping", None)

    if label_mapping is None:
        raise ValueError(
            f"SensorConfig for {info.sensor_name} does not contain label_mapping."
        )

    required_keys = {"invalid_pixel", "sealed_soil", "non_sealed_soil"}
    missing_keys = required_keys - set(label_mapping.keys())

    if missing_keys:
        raise ValueError(
            f"Incomplete label_mapping for {info.sensor_name}. "
            f"Missing: {missing_keys}"
        )

    raw_invalid = int(label_mapping["invalid_pixel"])
    raw_sealed = int(label_mapping["sealed_soil"])
    raw_non_sealed = int(label_mapping["non_sealed_soil"])

    valid_mask = label != raw_invalid

    valid_frac = float(valid_mask.mean())
    invalid_frac = 1.0 - valid_frac

    if invalid_frac > max_invalid_frac:
        return False

    if valid_frac < min_valid_frac:
        return False

    # Only real semantic classes are considered, not invalid pixels.
    semantic_mask = (label == raw_sealed) | (label == raw_non_sealed)
    semantic_classes = np.unique(label[semantic_mask])

    if len(semantic_classes) < min_valid_classes:
        return False

    return True


# ---------------------------------------------------------------------------
# Diagnostic utilities
# ---------------------------------------------------------------------------

def label_distribution(label: np.ndarray) -> dict:
    """
    Return the percentage distribution of label values.

    This is useful to inspect class imbalance between:

        0 = invalid_pixel
        1 = sealed_soil
        2 = non_sealed_soil

    Parameters
    ----------
    label :
        Original or remapped label array.

    Returns
    -------
    dict
        Mapping:

            label_value -> pixel_fraction
    """
    if label.ndim != 2:
        raise ValueError(
            f"label must have shape (H, W), received shape={label.shape}"
        )

    values, counts = np.unique(label, return_counts=True)
    total = label.size

    return {
        int(v): float(c) / float(total)
        for v, c in zip(values, counts)
    }


def check_image_range(x: np.ndarray) -> Tuple[float, float, float]:
    """
    Return min, max and mean values of an image.

    This is useful for debugging after normalization.

    Parameters
    ----------
    x :
        Image array with shape (C, H, W).

    Returns
    -------
    tuple of float
        Minimum, maximum and mean image values.
    """
    if x.ndim != 3:
        raise ValueError(f"x must have shape (C, H, W), received shape={x.shape}")

    return float(x.min()), float(x.max()), float(x.mean())