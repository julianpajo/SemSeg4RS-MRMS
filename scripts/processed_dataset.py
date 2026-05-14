"""
preprocessing/processed_dataset.py
----------------------------------

PyTorch dataset for data already preprocessed offline and stored as GeoTIFF files.

Each sample must contain:

    image: data/processed/crossearth/planetscope/train/images/tile_patch_000000.tif
    label: data/processed/crossearth/planetscope/train/labels/tile_patch_000000.tif
    sensor: planetscope
    model_name: crossearth
    source_image: optional
    source_label: optional
    row: optional
    col: optional
    crop_px: optional
    wavelengths: optional

Expected image GeoTIFF format
-----------------------------
    - multi-band
    - float32
    - rasterio shape: (C, H, W)
    - already normalized values, approximately in [0, 1]

Expected label GeoTIFF format
-----------------------------
    - single-band
    - uint8 or integer type
    - rasterio shape: (H, W)
    - values:
        0   = sealed_soil
        1   = non_sealed_soil
        255 = ignore_index

This Dataset does NOT perform:
    - cropping
    - band selection
    - normalization
    - resizing
    - label remapping

These operations have already been performed offline by:

    scripts/build_processed_dataset.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class ProcessedSegDataset(Dataset):
    """
    Dataset for images and labels already preprocessed offline as GeoTIFF files.

    This dataset only performs:

      - GeoTIFF reading with rasterio;
      - conversion to torch.Tensor;
      - optional transform application.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        transform=None,
        return_meta: bool = True,
    ):
        """
        Initialize the processed segmentation dataset.

        Parameters
        ----------
        samples :
            List of sample dictionaries. Each sample must contain at least
            "image" and "label".
        transform :
            Optional transform applied after reading and tensor conversion.
        return_meta :
            If True, return metadata such as file paths, sensor name, source
            files and crop coordinates.

        Raises
        ------
        ValueError
            If samples is empty or if a sample is missing required fields.
        FileNotFoundError
            If an image or label GeoTIFF file does not exist.
        """
        if len(samples) == 0:
            raise ValueError("samples is empty.")

        self.samples = samples
        self.transform = transform
        self.return_meta = return_meta

        for i, sample in enumerate(samples):
            if "image" not in sample:
                raise ValueError(f"Sample {i} does not contain 'image'.")

            if "label" not in sample:
                raise ValueError(f"Sample {i} does not contain 'label'.")

            image_path = Path(sample["image"])
            label_path = Path(sample["label"])

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image GeoTIFF not found: {image_path}"
                )

            if not label_path.exists():
                raise FileNotFoundError(
                    f"Label GeoTIFF not found: {label_path}"
                )

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Dataset length.
        """
        return len(self.samples)

    @staticmethod
    def _read_image(path: str | Path) -> np.ndarray:
        """
        Read a preprocessed multi-band image.

        Parameters
        ----------
        path :
            Path to the image GeoTIFF.

        Returns
        -------
        np.ndarray
            float32 image array with shape (C, H, W).

        Raises
        ------
        ValueError
            If the image does not have shape (C, H, W).
        """
        with rasterio.open(path) as src:
            image = src.read().astype(np.float32)

        if image.ndim != 3:
            raise ValueError(
                f"Image must have shape (C, H, W), received {image.shape}: {path}"
            )

        return image

    @staticmethod
    def _read_label(path: str | Path) -> np.ndarray:
        """
        Read a preprocessed single-band label.

        Parameters
        ----------
        path :
            Path to the label GeoTIFF.

        Returns
        -------
        np.ndarray
            int64 label array with shape (H, W).

        Raises
        ------
        ValueError
            If the label does not have shape (H, W).
        """
        with rasterio.open(path) as src:
            label = src.read(1).astype(np.int64)

        if label.ndim != 2:
            raise ValueError(
                f"Label must have shape (H, W), received {label.shape}: {path}"
            )

        return label

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Return one preprocessed dataset sample.

        Parameters
        ----------
        index :
            Dataset index.

        Returns
        -------
        dict
            Sample dictionary containing image tensor, label tensor, sensor name,
            wavelengths and optional metadata.

        Raises
        ------
        ValueError
            If image and label spatial shapes are inconsistent.
        """
        sample = self.samples[index]

        image_path = sample["image"]
        label_path = sample["label"]

        image = self._read_image(image_path)
        label = self._read_label(label_path)

        if image.shape[-2:] != label.shape[-2:]:
            raise ValueError(
                f"Inconsistent shapes: image={image.shape}, label={label.shape}. "
                f"image_path={image_path}, label_path={label_path}"
            )

        out = {
            "image": torch.from_numpy(image).float(),
            "label": torch.from_numpy(label).long(),
            "sensor": sample.get("sensor", "unknown"),
            "wavelengths": sample.get("wavelengths", None),
        }

        if self.return_meta:
            out["meta"] = {
                "image_path": image_path,
                "label_path": label_path,
                "sensor": sample.get("sensor", "unknown"),
                "model_name": sample.get("model_name", None),
                "source_image": sample.get("source_image", None),
                "source_label": sample.get("source_label", None),
                "row": sample.get("row", None),
                "col": sample.get("col", None),
                "crop_px": sample.get("crop_px", None),
            }

        if self.transform is not None:
            out = self.transform(out)

        return out

    def get_sensor_for_index(self, index: int) -> str:
        """
        Return the sensor associated with a dataset index.

        This method is used by SensorBatchSampler to create mono-sensor batches.

        Parameters
        ----------
        index :
            Dataset index.

        Returns
        -------
        str
            Sensor name associated with the sample.
        """
        return str(self.samples[index].get("sensor", "unknown"))

    def describe(self) -> Dict[str, Any]:
        """
        Return a compact summary of the processed dataset.

        Returns
        -------
        dict
            Dataset summary containing number of samples, sensors, sample counts
            by sensor, storage format and transform information.
        """
        sensors = sorted({s.get("sensor", "unknown") for s in self.samples})

        counts: Dict[str, int] = {}

        for sample in self.samples:
            sensor = sample.get("sensor", "unknown")
            counts[sensor] = counts.get(sensor, 0) + 1

        return {
            "num_samples": len(self.samples),
            "sensors": sensors,
            "sample_counts_by_sensor": counts,
            "preprocessed": True,
            "format": "geotiff",
            "has_transform": self.transform is not None,
        }