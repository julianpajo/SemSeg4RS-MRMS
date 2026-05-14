"""
preprocessing/dataset.py
------------------------

PyTorch dataset for multi-sensor remote-sensing semantic segmentation.

Responsibilities of this file:
  - read GeoTIFF images;
  - read GeoTIFF labels;
  - extract random crops for training;
  - extract regular grid crops for validation/test;
  - call preprocess_image_for_model();
  - call preprocess_label_for_model();
  - apply optional transforms;
  - return a single sample.

This file does not perform:
  - batching;
  - padding;
  - GPU transfer;
  - model forward logic.

Those responsibilities are handled in collate.py / trainer.py.

Expected sample format
----------------------
    {
        "image": ".../image.tif",
        "label": ".../label.tif",
        "sensor": "spot",
        "sensor_config": "configs/sensors/spot.yaml",  # optional but recommended
    }

If "sensor_config" is not provided, the dataset tries:

    configs/sensors/{sensor}.yaml
    configs/metadata/{sensor}.yaml
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

from configs.sensor_configs import read_sensor
from .preprocess import (
    IGNORE_INDEX,
    is_valid_patch,
    preprocess_image_for_model,
    preprocess_label_for_model,
)


# ---------------------------------------------------------------------------
# Raster utilities
# ---------------------------------------------------------------------------

def raster_size(path: str) -> Tuple[int, int]:
    """
    Return the spatial size of a raster.

    Parameters
    ----------
    path :
        Path to the raster file.

    Returns
    -------
    tuple of int
        Raster size as:

        height, width
    """
    with rasterio.open(path) as src:
        return src.height, src.width


def read_image_window(
    path: str,
    row: int,
    col: int,
    size: int,
) -> np.ndarray:
    """
    Read a square window from a multi-band raster.

    Parameters
    ----------
    path :
        Path to the image raster.
    row :
        Top-left row offset of the crop.
    col :
        Top-left column offset of the crop.
    size :
        Crop size in pixels.

    Returns
    -------
    np.ndarray
        Image array with shape (C, H, W).
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
    path: str,
    row: int,
    col: int,
    size: int,
) -> np.ndarray:
    """
    Read a square window from a single-band label raster.

    Parameters
    ----------
    path :
        Path to the label raster.
    row :
        Top-left row offset of the crop.
    col :
        Top-left column offset of the crop.
    size :
        Crop size in pixels.

    Returns
    -------
    np.ndarray
        Label array with shape (H, W), stored as int64.
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


def infer_sensor_config_path(sample: Dict[str, Any]) -> str:
    """
    Find the YAML sensor configuration file associated with a sample.

    Search priority
    ---------------
    1. sample["sensor_config"]
    2. configs/sensors/{sensor}.yaml
    3. configs/metadata/{sensor}.yaml

    Parameters
    ----------
    sample :
        Dataset sample dictionary.

    Returns
    -------
    str
        Path to the inferred sensor configuration file.

    Raises
    ------
    ValueError
        If neither "sensor_config" nor "sensor" is available.
    FileNotFoundError
        If no matching configuration file can be found.
    """
    if sample.get("sensor_config") is not None:
        return str(sample["sensor_config"])

    sensor = sample.get("sensor")

    if sensor is None:
        raise ValueError(
            "The sample must contain either 'sensor_config' or 'sensor'."
        )

    candidates = [
        Path("configs") / "sensors" / f"{sensor}.yaml",
        Path("configs") / "metadata" / f"{sensor}.yaml",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        f"Sensor configuration not found for sensor='{sensor}'. "
        f"Searched paths: {[str(p) for p in candidates]}"
    )


def compute_crop_size_px(
    image_path: str,
    sensor_info: Any,
    patch_size_px: Optional[int],
    patch_size_m: Optional[float],
) -> int:
    """
    Determine the crop size in pixels.

    Priority
    --------
    1. patch_size_px
    2. patch_size_m / sensor_info.gsd_m
    3. min(height, width), used as fallback

    Parameters
    ----------
    image_path :
        Path to the image raster.
    sensor_info :
        SensorConfig-like object containing at least gsd_m when patch_size_m is used.
    patch_size_px :
        Crop size in pixels. Takes priority if provided.
    patch_size_m :
        Crop size in meters. Used only if patch_size_px is None.

    Returns
    -------
    int
        Crop size in pixels.

    Raises
    ------
    ValueError
        If the provided crop size or GSD is invalid.
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


def validate_sample(sample: Dict[str, Any], idx: int) -> None:
    """
    Validate that a sample contains the minimum required fields.

    Required fields are:

      - image
      - label
      - sensor or sensor_config

    Parameters
    ----------
    sample :
        Sample dictionary to validate.
    idx :
        Sample index, used in error messages.

    Raises
    ------
    ValueError
        If required fields are missing.
    FileNotFoundError
        If the image or label file does not exist.
    """
    for key in ("image", "label"):
        if key not in sample:
            raise ValueError(
                f"Sample index {idx} does not contain the field '{key}'. "
                f"Received sample: {sample}"
            )

    if "sensor" not in sample and "sensor_config" not in sample:
        raise ValueError(
            f"Sample index {idx} must contain either 'sensor' or 'sensor_config'."
        )

    image_path = Path(sample["image"])
    label_path = Path(sample["label"])

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found for sample index {idx}: {image_path}"
        )

    if not label_path.exists():
        raise FileNotFoundError(
            f"Label not found for sample index {idx}: {label_path}"
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiSensorSegDataset(Dataset):
    """
    Dataset for multi-sensor semantic segmentation.

    Parameters
    ----------
    samples :
        List of dictionaries. Each sample must contain at least:

            image
            label
            sensor or sensor_config

    model_name :
        Model name handled by preprocess.py.

        Examples:

            "deeplabv3plus"
            "segformer_sae"
            "dofa"
            "crossearth"

        In the CrossEarth RGBNIR case, MODEL_REQUIREMENTS["crossearth"]["band_mode"]
        should be set to "rgbnir".

    split :
        Dataset split. Must be "train", "val", or "test".

    patch_size_px :
        Crop size in pixels before model-aware resizing.

    patch_size_m :
        Crop size in meters. Used only if patch_size_px is None.

    stride_px :
        Stride in pixels for validation/test crops.
        If None, stride = crop_size_px.

    max_invalid_frac :
        Maximum allowed fraction of invalid pixels in the raw label patch.

    min_valid_frac :
        Minimum required fraction of valid pixels in the raw label patch.

    min_valid_classes :
        Minimum number of semantically valid classes required in the patch.

    max_retries :
        Maximum number of attempts to find a valid training patch.

    remap_labels :
        If True, remap raw labels as:

            raw 0 -> 255
            raw 1 -> 0
            raw 2 -> 1

    ignore_index :
        Ignore index used for invalid pixels/classes during training.

    transform :
        Optional transform applied after preprocessing.

    return_meta :
        If True, return metadata such as file paths, crop coordinates and raw shapes.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        model_name: str,
        split: str = "train",
        patch_size_px: Optional[int] = None,
        patch_size_m: Optional[float] = None,
        stride_px: Optional[int] = None,
        max_invalid_frac: float = 0.9,
        min_valid_frac: float = 0.1,
        min_valid_classes: int = 1,
        max_retries: int = 20,
        remap_labels: bool = True,
        ignore_index: int = IGNORE_INDEX,
        transform=None,
        return_meta: bool = True,
    ):
        """
        Initialize the multi-sensor segmentation dataset.

        The constructor validates samples, stores configuration parameters and
        precomputes the validation/test crop grid when required.

        Raises
        ------
        ValueError
            If the split is invalid, samples is empty, or any sample is malformed.
        FileNotFoundError
            If any sample image or label path does not exist.
        """
        super().__init__()

        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"Invalid split: {split}. Use 'train', 'val', or 'test'."
            )

        if len(samples) == 0:
            raise ValueError("samples is empty.")

        for idx, sample in enumerate(samples):
            validate_sample(sample, idx)

        self.samples = samples
        self.model_name = model_name
        self.split = split

        self.patch_size_px = patch_size_px
        self.patch_size_m = patch_size_m
        self.stride_px = stride_px

        self.max_invalid_frac = max_invalid_frac
        self.min_valid_frac = min_valid_frac
        self.min_valid_classes = min_valid_classes
        self.max_retries = max_retries

        self.remap_labels = remap_labels
        self.ignore_index = ignore_index
        self.transform = transform
        self.return_meta = return_meta

        self._sensor_cache: Dict[int, Any] = {}

        self._items: Optional[List[Dict[str, int]]] = None

        if self.split in {"val", "test"}:
            self._items = self._build_grid_items()

    # ------------------------------------------------------------------
    # Public helpers used by samplers/collate
    # ------------------------------------------------------------------

    def get_sample_index(self, dataset_index: int) -> int:
        """
        Return the original sample index corresponding to a Dataset index.

        For training:

            dataset_index == sample_index

        For validation/test:

            dataset_index refers to a grid patch, so this method maps it back
            to the original sample index.

        Parameters
        ----------
        dataset_index :
            Index used by the PyTorch Dataset.

        Returns
        -------
        int
            Original sample index.
        """
        if self._items is None:
            return int(dataset_index)

        return int(self._items[dataset_index]["sample_idx"])

    def get_sensor_for_index(self, dataset_index: int) -> str:
        """
        Return the sensor associated with a Dataset index.

        This method is used by SensorBatchSampler, so the sampler does not need
        to know internal attributes such as _items.

        Parameters
        ----------
        dataset_index :
            Index used by the PyTorch Dataset.

        Returns
        -------
        str
            Sensor name associated with the item.
        """
        sample_idx = self.get_sample_index(dataset_index)
        sample = self.samples[sample_idx]

        return str(sample.get("sensor", "unknown"))

    def describe(self) -> Dict[str, Any]:
        """
        Return a compact summary of the dataset configuration and content.

        Returns
        -------
        dict
            Dataset summary including split, number of samples, number of items,
            sensors, crop settings and label settings.
        """
        return {
            "num_samples": len(self.samples),
            "num_dataset_items": len(self),
            "split": self.split,
            "model_name": self.model_name,
            "patch_size_px": self.patch_size_px,
            "patch_size_m": self.patch_size_m,
            "stride_px": self.stride_px,
            "sensors": self.get_sensors(),
            "sample_counts_by_sensor": self.get_sample_counts_by_sensor(),
            "remap_labels": self.remap_labels,
            "ignore_index": self.ignore_index,
            "has_transform": self.transform is not None,
        }

    def get_sensors(self) -> List[str]:
        """
        Return the sorted list of sensors present in the dataset.

        Returns
        -------
        list of str
            Sorted sensor names.
        """
        return sorted({str(s.get("sensor", "unknown")) for s in self.samples})

    def get_sample_counts_by_sensor(self) -> Dict[str, int]:
        """
        Count the number of original samples for each sensor.

        Returns
        -------
        dict
            Mapping from sensor name to number of samples.
        """
        counts: Dict[str, int] = {}

        for sample in self.samples:
            sensor = str(sample.get("sensor", "unknown"))
            counts[sensor] = counts.get(sensor, 0) + 1

        return counts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_sensor_info(self, sample_idx: int):
        """
        Read and cache the SensorConfig associated with a sample.

        Parameters
        ----------
        sample_idx :
            Original sample index.

        Returns
        -------
        Any
            SensorConfig object returned by read_sensor().
        """
        if sample_idx in self._sensor_cache:
            return self._sensor_cache[sample_idx]

        sample = self.samples[sample_idx]
        image_path = sample["image"]
        yaml_path = infer_sensor_config_path(sample)

        info = read_sensor(image_path, yaml_path)

        self._sensor_cache[sample_idx] = info

        return info

    def _get_crop_size_px(self, sample_idx: int) -> int:
        """
        Compute the crop size in pixels for a specific sample.

        The crop size is clipped to the raster dimensions, so it never exceeds
        min(height, width).

        Parameters
        ----------
        sample_idx :
            Original sample index.

        Returns
        -------
        int
            Crop size in pixels.
        """
        sample = self.samples[sample_idx]
        info = self._get_sensor_info(sample_idx)

        h, w = raster_size(sample["image"])

        crop_px = compute_crop_size_px(
            image_path=sample["image"],
            sensor_info=info,
            patch_size_px=self.patch_size_px,
            patch_size_m=self.patch_size_m,
        )

        return int(min(crop_px, h, w))

    def _build_grid_items(self) -> List[Dict[str, int]]:
        """
        Create the list of regular grid crops for validation/test.

        Returns
        -------
        list of dict
            Each item contains:

            sample_idx :
                Original sample index.
            row :
                Crop top-left row.
            col :
                Crop top-left column.
            crop_px :
                Crop size in pixels.
        """
        items: List[Dict[str, int]] = []

        for sample_idx, sample in enumerate(self.samples):
            h, w = raster_size(sample["image"])
            crop_px = self._get_crop_size_px(sample_idx)

            stride = self.stride_px if self.stride_px is not None else crop_px

            if stride <= 0:
                raise ValueError(f"stride_px must be > 0, received {stride}")

            max_row = max(0, h - crop_px)
            max_col = max(0, w - crop_px)

            rows = list(range(0, max_row + 1, stride))
            cols = list(range(0, max_col + 1, stride))

            if not rows:
                rows = [0]

            if not cols:
                cols = [0]

            if rows[-1] != max_row:
                rows.append(max_row)

            if cols[-1] != max_col:
                cols.append(max_col)

            for row in rows:
                for col in cols:
                    items.append(
                        {
                            "sample_idx": int(sample_idx),
                            "row": int(row),
                            "col": int(col),
                            "crop_px": int(crop_px),
                        }
                    )

        return items

    def _sample_random_window(self, sample_idx: int) -> Dict[str, int]:
        """
        Select a random valid crop for training.

        The method tries up to max_retries random windows and returns the first
        valid one according to is_valid_patch(). If no valid patch is found, it
        returns the last sampled window.

        Parameters
        ----------
        sample_idx :
            Original sample index.

        Returns
        -------
        dict
            Crop item containing sample_idx, row, col and crop_px.
        """
        sample = self.samples[sample_idx]
        h, w = raster_size(sample["image"])
        crop_px = self._get_crop_size_px(sample_idx)

        last_item = {
            "sample_idx": int(sample_idx),
            "row": 0,
            "col": 0,
            "crop_px": int(crop_px),
        }

        for _ in range(self.max_retries):
            row = random.randint(0, max(0, h - crop_px))
            col = random.randint(0, max(0, w - crop_px))

            label_raw = read_label_window(
                path=sample["label"],
                row=row,
                col=col,
                size=crop_px,
            )

            last_item = {
                "sample_idx": int(sample_idx),
                "row": int(row),
                "col": int(col),
                "crop_px": int(crop_px),
            }

            info = self._get_sensor_info(sample_idx)

            if is_valid_patch(
                label=label_raw,
                info=info,
                max_invalid_frac=self.max_invalid_frac,
                min_valid_frac=self.min_valid_frac,
                min_valid_classes=self.min_valid_classes,
            ):
                return last_item

        return last_item

    def _load_raw_pair(self, item: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read the raw image and raw label corresponding to a crop item.

        Parameters
        ----------
        item :
            Crop item containing sample_idx, row, col and crop_px.

        Returns
        -------
        tuple
            Tuple containing:

            image_raw :
                Raw image array with shape (C, H, W).
            label_raw :
                Raw label array with shape (H, W).
        """
        sample = self.samples[item["sample_idx"]]

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

        return image_raw, label_raw

    def _process_raw_pair(
        self,
        image_raw: np.ndarray,
        label_raw: np.ndarray,
        item: Dict[str, int],
    ) -> Dict[str, Any]:
        """
        Apply preprocessing and build the final dataset sample.

        Parameters
        ----------
        image_raw :
            Raw image crop with shape (C, H, W).
        label_raw :
            Raw label crop with shape (H, W).
        item :
            Crop item containing sample index and crop coordinates.

        Returns
        -------
        dict
            Final sample dictionary containing image tensor, label tensor,
            sensor name, wavelengths and optional metadata.
        """
        sample_idx = item["sample_idx"]
        sample = self.samples[sample_idx]
        info = self._get_sensor_info(sample_idx)

        image_np, wavelengths = preprocess_image_for_model(
            x=image_raw,
            info=info,
            model_name=self.model_name,
        )

        label_np = preprocess_label_for_model(
            y=label_raw,
            info=info,
            model_name=self.model_name,
            remap=self.remap_labels,
            ignore_index=self.ignore_index,
        )

        sensor_name = str(
            sample.get("sensor", getattr(info, "sensor_name", "unknown"))
        )

        out: Dict[str, Any] = {
            "image": torch.from_numpy(image_np).float(),
            "label": torch.from_numpy(label_np).long(),
            "sensor": sensor_name,
            "wavelengths": [float(w) for w in wavelengths],
        }

        if self.return_meta:
            out["meta"] = {
                "image_path": sample["image"],
                "label_path": sample["label"],
                "sample_idx": int(sample_idx),
                "row": int(item["row"]),
                "col": int(item["col"]),
                "crop_px": int(item["crop_px"]),
                "model_name": self.model_name,
                "raw_image_shape": tuple(image_raw.shape),
                "raw_label_shape": tuple(label_raw.shape),
            }

        if self.transform is not None:
            out = self.transform(out)

        return out

    # ------------------------------------------------------------------
    # PyTorch API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        For training, this corresponds to the number of original samples.
        For validation/test, this corresponds to the number of grid crops.

        Returns
        -------
        int
            Dataset length.
        """
        if self._items is None:
            return len(self.samples)

        return len(self._items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Return one preprocessed dataset sample.

        For training, a random crop is sampled from the selected image.
        For validation/test, a deterministic grid crop is used.

        Parameters
        ----------
        index :
            Dataset index.

        Returns
        -------
        dict
            Preprocessed sample containing image, label, sensor, wavelengths
            and optional metadata.
        """
        if self._items is None:
            item = self._sample_random_window(sample_idx=index)
        else:
            item = self._items[index]

        image_raw, label_raw = self._load_raw_pair(item)

        return self._process_raw_pair(
            image_raw=image_raw,
            label_raw=label_raw,
            item=item,
        )