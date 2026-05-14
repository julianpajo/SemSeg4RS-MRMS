"""
preprocessing/collate.py
------------------------

Collate functions and adapters for multi-sensor semantic segmentation.

Responsibilities of this file:
  - transform a list of samples into a batch;
  - stack images and labels;
  - handle optional channel padding for DOFA;
  - provide SensorBatchSampler without accessing internal Dataset details;
  - prepare batches for the model forward pass.

This file does not perform:
  - raster reading;
  - cropping;
  - image/label preprocessing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Iterator

import numpy as np
import torch
from torch.utils.data import Sampler


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _assert_same_shape(tensors: List[torch.Tensor], key: str) -> None:
    """
    Check that all tensors in a list have the same shape.

    Parameters
    ----------
    tensors :
        List of tensors to check.
    key :
        Field name used in the error message.

    Raises
    ------
    ValueError
        If tensors have different shapes.
    """
    shapes = [tuple(t.shape) for t in tensors]

    if len(set(shapes)) != 1:
        raise ValueError(
            f"Tensors '{key}' in the batch have different shapes: {shapes}"
        )


def _collect_non_tensor_fields(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collect all non-tensor fields into lists.

    Examples
    --------
        sensor      -> list[str]
        wavelengths -> list[list[float]]
        meta        -> list[dict]

    This avoids duplicating metadata field handling inside each collate
    function.

    Parameters
    ----------
    batch :
        List of dataset samples.

    Returns
    -------
    dict
        Dictionary containing all non-tensor fields collected as lists.
    """
    out: Dict[str, Any] = {}

    tensor_keys = {"image", "label"}

    keys = set()
    for sample in batch:
        keys.update(sample.keys())

    for key in sorted(keys):
        if key in tensor_keys:
            continue

        out[key] = [sample.get(key) for sample in batch]

    return out


# ---------------------------------------------------------------------------
# Standard collate
# ---------------------------------------------------------------------------

def segmentation_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Standard collate function for models with a fixed number of input channels.

    Requirements
    ------------
    All samples in the batch must have:

      - images with the same shape (C, H, W);
      - labels with the same shape (H, W).

    Parameters
    ----------
    batch :
        List of dataset samples.

    Returns
    -------
    dict
        Batch dictionary containing:

        image :
            Tensor with shape (B, C, H, W).
        label :
            Tensor with shape (B, H, W).
        other fields :
            Non-tensor metadata fields collected into lists.
    """
    if len(batch) == 0:
        raise ValueError("Empty batch.")

    images = [sample["image"] for sample in batch]
    labels = [sample["label"] for sample in batch]

    _assert_same_shape(images, key="image")
    _assert_same_shape(labels, key="label")

    out = {
        "image": torch.stack(images, dim=0).float(),
        "label": torch.stack(labels, dim=0).long(),
    }

    out.update(_collect_non_tensor_fields(batch))

    return out


# Backward-compatible name used by older code.
mono_sensor_collate = segmentation_collate


# ---------------------------------------------------------------------------
# DOFA collate with channel padding
# ---------------------------------------------------------------------------

def dofa_pad_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for models that can handle batches with different band counts.

    This function pads the channel dimension C up to the maximum number of
    channels in the current batch.

    Parameters
    ----------
    batch :
        List of dataset samples.

    Returns
    -------
    dict
        Batch dictionary containing:

        image :
            Tensor with shape (B, C_max, H, W).
        label :
            Tensor with shape (B, H, W).
        band_mask :
            Boolean tensor with shape (B, C_max), where True marks valid bands.
        other fields :
            Non-tensor metadata fields collected into lists.

    Notes
    -----
    Use this function only if the model can interpret band_mask, or if padding
    is explicitly handled inside the forward pass.
    """
    if len(batch) == 0:
        raise ValueError("Empty batch.")

    images = [sample["image"].float() for sample in batch]
    labels = [sample["label"].long() for sample in batch]

    _assert_same_shape(labels, key="label")

    spatial_shapes = [(img.shape[-2], img.shape[-1]) for img in images]

    if len(set(spatial_shapes)) != 1:
        raise ValueError(
            f"Images have different spatial dimensions: {spatial_shapes}"
        )

    h, w = spatial_shapes[0]
    c_max = max(img.shape[0] for img in images)

    padded_images = []
    band_masks = []

    for img in images:
        c = img.shape[0]

        padded = torch.zeros(
            (c_max, h, w),
            dtype=img.dtype,
        )
        padded[:c] = img

        mask = torch.zeros(
            (c_max,),
            dtype=torch.bool,
        )
        mask[:c] = True

        padded_images.append(padded)
        band_masks.append(mask)

    out = {
        "image": torch.stack(padded_images, dim=0),
        "label": torch.stack(labels, dim=0),
        "band_mask": torch.stack(band_masks, dim=0),
    }

    out.update(_collect_non_tensor_fields(batch))

    return out


# ---------------------------------------------------------------------------
# Sensor batch sampler
# ---------------------------------------------------------------------------

class SensorBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that builds mono-sensor batches.

    This sampler no longer accesses dataset._grid_items or any other internal
    Dataset detail. It only uses:

        dataset.get_sensor_for_index(index)

    This keeps collate.py independent from the internal Dataset structure.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        """
        Initialize the sensor-aware batch sampler.

        Parameters
        ----------
        dataset :
            Dataset object implementing get_sensor_for_index(index).
        batch_size :
            Number of samples per batch.
        shuffle :
            If True, shuffle indices within each sensor group and shuffle the
            final batch order.
        drop_last :
            If True, discard incomplete batches.

        Raises
        ------
        ValueError
            If batch_size is not strictly positive.
        TypeError
            If the dataset does not implement get_sensor_for_index(index).
        """
        if batch_size <= 0:
            raise ValueError(
                f"batch_size must be > 0, received {batch_size}"
            )

        if not hasattr(dataset, "get_sensor_for_index"):
            raise TypeError(
                "The dataset must implement get_sensor_for_index(index)."
            )

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

        self.groups = self._build_groups()

    def _build_groups(self) -> Dict[str, List[int]]:
        """
        Group dataset indices by sensor name.

        Returns
        -------
        dict
            Dictionary mapping each sensor name to the list of dataset indices
            associated with that sensor.
        """
        groups: Dict[str, List[int]] = {}

        for idx in range(len(self.dataset)):
            sensor = self.dataset.get_sensor_for_index(idx)
            groups.setdefault(sensor, []).append(idx)

        return groups

    def __iter__(self) -> Iterator[List[int]]:
        """
        Yield batches of indices grouped by sensor.

        Yields
        ------
        list of int
            Dataset indices belonging to the same sensor.
        """
        batches: List[List[int]] = []

        for _, indices in self.groups.items():
            idx = indices.copy()

            if self.shuffle:
                np.random.shuffle(idx)

            for start in range(0, len(idx), self.batch_size):
                batch = idx[start:start + self.batch_size]

                if self.drop_last and len(batch) < self.batch_size:
                    continue

                batches.append(batch)

        if self.shuffle:
            np.random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        """
        Return the number of batches produced by the sampler.

        Returns
        -------
        int
            Total number of mono-sensor batches.
        """
        total = 0

        for indices in self.groups.values():
            n = len(indices) // self.batch_size

            if not self.drop_last and len(indices) % self.batch_size != 0:
                n += 1

            total += n

        return total


# ---------------------------------------------------------------------------
# Model adapter
# ---------------------------------------------------------------------------

class ModelAdapter:
    """
    Prepare a batch for the model forward pass.

    Standard models
    ---------------
        logits = model(image)

    DOFA models
    -----------
        logits = model(image, wavelengths=...)

    Optionally:

        logits = model(image, wavelengths=..., band_mask=...)
    """

    WAVELENGTH_MODELS = {"dofa"}

    def __init__(
        self,
        model_type: str,
        device: torch.device,
        use_band_mask: bool = False,
    ):
        """
        Initialize the model adapter.

        Parameters
        ----------
        model_type :
            Model family/type name. Used to determine whether wavelengths must
            be passed to the forward call.
        device :
            Torch device where tensors should be moved.
        use_band_mask :
            If True, pass band_mask to wavelength-aware models when available.
        """
        self.model_type = model_type.lower()
        self.device = device
        self.use_band_mask = use_band_mask

    def prepare_batch(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Move tensors to the target device and prepare model keyword arguments.

        Parameters
        ----------
        batch :
            Batch dictionary produced by a collate function.

        Returns
        -------
        tuple
            Tuple containing:

            image :
                Input image tensor moved to the selected device.
            label :
                Label tensor moved to the selected device.
            kwargs :
                Extra keyword arguments to pass to the model forward method.
        """
        image = batch["image"].to(self.device, non_blocking=True)
        label = batch["label"].to(self.device, non_blocking=True)

        kwargs: Dict[str, Any] = {}

        if self.model_type in self.WAVELENGTH_MODELS:
            kwargs["wavelengths"] = batch.get("wavelengths")

            if self.use_band_mask and "band_mask" in batch:
                kwargs["band_mask"] = batch["band_mask"].to(
                    self.device,
                    non_blocking=True,
                )

        return image, label, kwargs

    def forward(
        self,
        model: torch.nn.Module,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run a model forward pass using a prepared batch.

        Parameters
        ----------
        model :
            PyTorch model to evaluate.
        batch :
            Batch dictionary produced by a collate function.

        Returns
        -------
        tuple
            Tuple containing:

            logits :
                Model output logits.
            label :
                Ground-truth label tensor moved to the selected device.
        """
        image, label, kwargs = self.prepare_batch(batch)
        logits = model(image, **kwargs)

        return logits, label