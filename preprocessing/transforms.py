"""
preprocessing/transforms.py
---------------------------

Data augmentation transformations for semantic segmentation.

Expected input
--------------
    image : torch.Tensor or np.ndarray, shape (C, H, W)
    label : torch.Tensor or np.ndarray, shape (H, W)

Geometric transformations are applied to both image and label.
Radiometric transformations are applied only to image.

Notes
-----
These transformations assume that the image has already been normalized
approximately to the [0, 1] range.
"""

from __future__ import annotations

import random
from typing import Dict

import numpy as np
import torch


def _to_tensor_image(x):
    """
    Convert an image array/tensor to a float torch.Tensor.

    Parameters
    ----------
    x :
        Image as np.ndarray or torch.Tensor with shape (C, H, W).

    Returns
    -------
    torch.Tensor
        Float tensor with shape (C, H, W).
    """
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return x.float()


def _to_tensor_label(y):
    """
    Convert a label array/tensor to a long torch.Tensor.

    Parameters
    ----------
    y :
        Label as np.ndarray or torch.Tensor with shape (H, W).

    Returns
    -------
    torch.Tensor
        Long tensor with shape (H, W).
    """
    if isinstance(y, np.ndarray):
        return torch.from_numpy(y).long()
    return y.long()


class SegmentationTrainTransform:
    """
    Simple training augmentation for semantic segmentation.

    Transformations
    ---------------
      - horizontal flip;
      - vertical flip;
      - random rotations by 0/90/180/270 degrees;
      - brightness jitter;
      - contrast jitter;
      - light Gaussian noise.

    Parameters
    ----------
    p_hflip :
        Probability of applying horizontal flip.
    p_vflip :
        Probability of applying vertical flip.
    p_rotate :
        Probability of applying a random 90-degree rotation.
    brightness :
        Maximum brightness range.
        Example: 0.05 means additive factor in [-0.05, +0.05].
    contrast :
        Contrast range.
        Example: 0.10 means multiplicative factor in [0.90, 1.10].
    noise_std :
        Maximum standard deviation of Gaussian noise.
    """

    def __init__(
        self,
        p_hflip: float = 0.5,
        p_vflip: float = 0.5,
        p_rotate: float = 0.5,
        brightness: float = 0.05,
        contrast: float = 0.10,
        noise_std: float = 0.01,
    ):
        """
        Initialize the training transform.

        Parameters
        ----------
        p_hflip :
            Probability of horizontal flip.
        p_vflip :
            Probability of vertical flip.
        p_rotate :
            Probability of random 90-degree rotation.
        brightness :
            Maximum additive brightness perturbation.
        contrast :
            Maximum contrast perturbation around 1.0.
        noise_std :
            Maximum Gaussian noise standard deviation.
        """
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rotate = p_rotate
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std

    def __call__(self, sample: Dict) -> Dict:
        """
        Apply training augmentations to a sample.

        Parameters
        ----------
        sample :
            Sample dictionary containing "image" and "label".

        Returns
        -------
        dict
            Augmented sample with image as float tensor and label as long tensor.
        """
        image = _to_tensor_image(sample["image"])
        label = _to_tensor_label(sample["label"])

        # Horizontal flip over W.
        if random.random() < self.p_hflip:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[1])

        # Vertical flip over H.
        if random.random() < self.p_vflip:
            image = torch.flip(image, dims=[1])
            label = torch.flip(label, dims=[0])

        # 90-degree rotations.
        if random.random() < self.p_rotate:
            k = random.randint(0, 3)
            image = torch.rot90(image, k=k, dims=[1, 2])
            label = torch.rot90(label, k=k, dims=[0, 1])

        # Brightness jitter.
        if self.brightness > 0:
            delta = random.uniform(-self.brightness, self.brightness)
            image = image + delta

        # Contrast jitter.
        if self.contrast > 0:
            factor = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
            mean = image.mean(dim=(1, 2), keepdim=True)
            image = (image - mean) * factor + mean

        # Gaussian noise.
        if self.noise_std > 0:
            std = random.uniform(0.0, self.noise_std)
            image = image + torch.randn_like(image) * std

        image = torch.clamp(image, 0.0, 1.0)

        sample = dict(sample)
        sample["image"] = image.float()
        sample["label"] = label.long()

        return sample


class SegmentationEvalTransform:
    """
    Evaluation transform for validation/test.

    No augmentation is applied. This transform only ensures that image and label
    have the correct tensor dtypes.
    """

    def __call__(self, sample: Dict) -> Dict:
        """
        Convert image and label to tensors with the correct dtypes.

        Parameters
        ----------
        sample :
            Sample dictionary containing "image" and "label".

        Returns
        -------
        dict
            Sample with image as float tensor and label as long tensor.
        """
        sample = dict(sample)
        sample["image"] = _to_tensor_image(sample["image"]).float()
        sample["label"] = _to_tensor_label(sample["label"]).long()
        return sample


class NormalizeWithStats:
    """
    Optional standardization with dataset mean/std.

    Use this after normalize() if each band should be transformed as:

        (x - mean) / std

    Parameters
    ----------
    mean :
        Per-channel mean values.
    std :
        Per-channel standard deviation values.
    eps :
        Small value added to std to avoid division by zero.
    """

    def __init__(self, mean, std, eps: float = 1e-6):
        """
        Initialize the standardization transform.

        Parameters
        ----------
        mean :
            Per-channel mean values.
        std :
            Per-channel standard deviation values.
        eps :
            Numerical stability constant.
        """
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
        self.eps = eps

    def __call__(self, sample: Dict) -> Dict:
        """
        Apply per-channel standardization to the image.

        Parameters
        ----------
        sample :
            Sample dictionary containing "image".

        Returns
        -------
        dict
            Sample with standardized image.
        """
        image = _to_tensor_image(sample["image"])

        if image.shape[0] != self.mean.shape[0]:
            raise ValueError(
                f"Inconsistent number of channels: image C={image.shape[0]}, "
                f"mean/std C={self.mean.shape[0]}"
            )

        sample = dict(sample)
        sample["image"] = (image - self.mean) / (self.std + self.eps)

        return sample