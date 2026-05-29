"""
Common configuration blocks and shared factory utilities for training scripts.

This module centralizes configuration components that are reused across all
model-specific training configuration files. It defines global project paths,
Lightning Storage locations, registry defaults, and immutable dataclass blocks
for paths, registry metadata, task settings, dataloader parameters, and runtime
options.

Model-specific configuration modules should import these common blocks and
compose them with their own model and optimization dataclasses.

The two factory helpers provided here are:

- :func:`build_common_paths`, which creates the standard project, data, chunk,
  and output paths for a given model/dataset pair.
- :func:`build_registry`, which creates the default Lightning AI model registry
  metadata for a given model/dataset pair.

Expected optimized dataset layout
---------------------------------
Optimized LitData chunks are expected to be shared across models and organized
only by dataset and split:

    data/optimized/<dataset_name>/<split>/

For example:

    data/optimized/planetscope/train/
    data/optimized/planetscope/val/

Model outputs are instead organized by model and dataset:

    models/<model_name>/<dataset_name>/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ============================================================
# Global paths
# ============================================================

PROJECT_ROOT = Path("/teamspace/studios/this_studio")
LIGHTNING_STORAGE_ROOT = Path("/teamspace/lightning_storage/pkq003_sensor_agnostic")

DATA_ROOT = LIGHTNING_STORAGE_ROOT / "data"
OUTPUT_ROOT = LIGHTNING_STORAGE_ROOT / "models"


# ============================================================
# Registry defaults
# ============================================================

DEFAULT_ORGANIZATION = "maurosylos"
DEFAULT_TEAMSPACE = "planetek-geoai"


# ============================================================
# Common config blocks
# ============================================================


@dataclass(frozen=True)
class PathConfig:
    """
    Filesystem paths used by a training run.

    Attributes
    ----------
    project_root:
        Root directory of the active Lightning Studio project.
    data_root:
        Root directory containing datasets and optimized LitData chunks.
    train_chunks:
        Path to the optimized LitData chunks used for the training split.
    val_chunks:
        Path to the optimized LitData chunks used for the validation split.
    output_dir:
        Directory where model-specific outputs, logs, and checkpoints are
        written.
    """

    project_root: Path
    data_root: Path
    train_chunks: Path
    val_chunks: Path
    output_dir: Path


@dataclass(frozen=True)
class RegistryConfig:
    """
    Lightning AI registry metadata.

    Attributes
    ----------
    organization:
        Lightning AI organization name.
    teamspace:
        Lightning AI teamspace name.
    model_registry_name:
        Name used to register or identify the trained model artifact.
    """

    organization: str
    teamspace: str
    model_registry_name: str


@dataclass(frozen=True)
class TaskConfig:
    """
    Semantic segmentation task configuration.

    Attributes
    ----------
    num_classes:
        Number of semantic classes predicted by the model, excluding ignored
        pixels.
    ignore_index:
        Label value ignored by the loss function and metric computation.
    in_channels:
        Number of input image channels. Set to ``None`` for sensor-agnostic
        models that infer or handle input channels internally.
    class_weights:
        Optional per-class weights used by weighted loss functions.
    """

    num_classes: int
    ignore_index: int
    in_channels: int | None = None
    class_weights: list[float] | None = None


@dataclass(frozen=True)
class DataLoaderConfig:
    """
    Dataloader and effective batch-size configuration.

    Attributes
    ----------
    batch_size:
        Per-device batch size used by the dataloader.
    num_workers:
        Number of worker processes used for data loading.
    gradient_accumulation:
        Number of gradient accumulation steps used by the trainer.
    """

    batch_size: int
    num_workers: int
    gradient_accumulation: int


@dataclass(frozen=True)
class RuntimeConfig:
    """
    Runtime configuration for PyTorch Lightning training.

    Attributes
    ----------
    epochs:
        Maximum number of training epochs.
    log_every_n_steps:
        Logging frequency expressed in training steps.
    patience:
        Early stopping patience, usually measured in validation checks.
    matmul_precision:
        PyTorch float32 matrix multiplication precision setting.
    """

    epochs: int
    log_every_n_steps: int
    patience: int
    matmul_precision: str = "high"


def build_common_paths(
    model_name: str,
    dataset_name: str,
) -> PathConfig:
    """
    Build the common project, dataset, chunk, and output paths.

    Parameters
    ----------
    model_name:
        Internal model identifier used to organize output directories.
    dataset_name:
        Dataset identifier used to locate optimized LitData chunks.

    Returns
    -------
    PathConfig
        Immutable path configuration for the selected model/dataset pair.

    Notes
    -----
    Optimized LitData chunks are dataset-level assets shared across models.
    Therefore, they are expected under:

        data/optimized/<dataset_name>/<split>/

    and not under:

        data/optimized/<model_name>/<dataset_name>/<split>/

    Model outputs are model-specific and are written under:

        models/<model_name>/<dataset_name>/
    """

    return PathConfig(
        project_root=PROJECT_ROOT,
        data_root=DATA_ROOT,
        train_chunks=DATA_ROOT / "optimized" / dataset_name / "train",
        val_chunks=DATA_ROOT / "optimized" / dataset_name / "val",
        output_dir=OUTPUT_ROOT / model_name / dataset_name,
    )


def build_registry(
    model_name: str,
    dataset_name: str,
    organization: str = DEFAULT_ORGANIZATION,
    teamspace: str = DEFAULT_TEAMSPACE,
) -> RegistryConfig:
    """
    Build the Lightning AI registry configuration.

    Parameters
    ----------
    model_name:
        Internal model identifier used as part of the registry name.
    dataset_name:
        Dataset identifier used as part of the registry name.
    organization:
        Lightning AI organization name. Defaults to
        :data:`DEFAULT_ORGANIZATION`.
    teamspace:
        Lightning AI teamspace name. Defaults to :data:`DEFAULT_TEAMSPACE`.

    Returns
    -------
    RegistryConfig
        Immutable registry configuration containing organization, teamspace,
        and model registry name.
    """

    return RegistryConfig(
        organization=organization,
        teamspace=teamspace,
        model_registry_name=f"{model_name}-{dataset_name}",
    )