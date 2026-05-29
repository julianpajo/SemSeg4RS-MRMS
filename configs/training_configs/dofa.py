"""
Configuration factory for DOFA semantic segmentation experiments.

This module defines the complete training configuration used by the DOFA model
within the project training pipeline. It combines common configuration blocks
shared across models, such as paths, registry metadata, task settings,
dataloader parameters, and runtime options, with DOFA-specific model and
optimization parameters.

The main entry point is :func:`get_config`, which builds a fully populated
:class:`DOFAConfig` instance for a selected dataset.

Typical usage
-------------
>>> from configs.training_configs.dofa import get_config
>>> cfg = get_config(dataset_name="planetscope")
>>> cfg.model.variant
'base'
"""

from __future__ import annotations

from dataclasses import dataclass

from configs.training_configs.common import (
    DataLoaderConfig,
    PathConfig,
    RegistryConfig,
    RuntimeConfig,
    TaskConfig,
    build_common_paths,
    build_registry,
)


# ============================================================
# Model-specific config blocks
# ============================================================


@dataclass(frozen=True)
class DOFAModelConfig:
    """
    Model-specific configuration for the DOFA architecture.

    Attributes
    ----------
    variant:
        DOFA backbone/model size identifier, for example ``"base"``.
    decoder:
        Decoder head used for semantic segmentation, for example ``"mla"``.
    use_pretrained:
        Whether to initialize the DOFA backbone from pretrained weights.
    freeze_backbone:
        Whether to freeze the DOFA backbone during training and update only the
        segmentation decoder or task-specific layers.
    """

    variant: str
    decoder: str
    use_pretrained: bool
    freeze_backbone: bool


@dataclass(frozen=True)
class DOFAOptimConfig:
    """
    Optimization configuration for DOFA training.

    Attributes
    ----------
    lr:
        Learning rate used by the optimizer.
    weight_decay:
        Weight decay coefficient used by the optimizer.
    scheduler_interval:
        Scheduler stepping interval, usually ``"step"`` or ``"epoch"``.
    """

    lr: float
    weight_decay: float
    scheduler_interval: str


@dataclass(frozen=True)
class DOFAConfig:
    """
    Complete experiment configuration for DOFA training.

    This dataclass aggregates all configuration blocks required to instantiate
    the dataset pipeline, model, optimizer, logger/registry integration, and
    Lightning runtime behavior.

    Attributes
    ----------
    dataset_name:
        Name of the dataset split family used for the experiment.
    model_name:
        Internal model identifier used for paths, logging, and registry naming.
    experiment_name:
        Human-readable experiment name, typically derived from model and dataset.
    paths:
        Common filesystem and storage paths used by the training script.
    registry:
        Lightning AI registry configuration for experiment/model tracking.
    task:
        Semantic segmentation task configuration, including class count, ignore
        index, and input channel settings.
    data:
        Dataloader configuration, including batch size, workers, and gradient
        accumulation.
    runtime:
        Runtime configuration for training epochs, logging cadence, patience,
        and precision settings.
    model:
        DOFA-specific architectural configuration.
    optim:
        DOFA-specific optimization configuration.
    """

    dataset_name: str
    model_name: str
    experiment_name: str

    paths: PathConfig
    registry: RegistryConfig
    task: TaskConfig
    data: DataLoaderConfig
    runtime: RuntimeConfig
    model: DOFAModelConfig
    optim: DOFAOptimConfig


# ============================================================
# Factory
# ============================================================


def get_config(dataset_name: str = "planetscope") -> DOFAConfig:
    """
    Build the default DOFA training configuration.

    Parameters
    ----------
    dataset_name:
        Dataset identifier used to derive paths, registry names, and the
        experiment name. Defaults to ``"planetscope"``.

    Returns
    -------
    DOFAConfig
        Fully populated immutable configuration object for a DOFA training run.

    Notes
    -----
    DOFA is sensor-agnostic with respect to input spectral channels; therefore
    ``TaskConfig.in_channels`` is set to ``None`` by default instead of using a
    fixed number of input bands.

    The default configuration initializes the DOFA backbone from pretrained
    weights and freezes it, training only the segmentation head or other
    task-specific parameters controlled by the downstream Lightning module.
    """

    model_name = "dofa"
    experiment_name = f"{model_name}-{dataset_name}"

    return DOFAConfig(
        dataset_name=dataset_name,
        model_name=model_name,
        experiment_name=experiment_name,
        paths=build_common_paths(
            model_name=model_name,
            dataset_name=dataset_name,
        ),
        registry=build_registry(
            model_name=model_name,
            dataset_name=dataset_name,
            organization="maurosylos",
            teamspace="planetek-geoai",
        ),
        task=TaskConfig(
            num_classes=2,
            ignore_index=255,
            in_channels=None,
        ),
        model=DOFAModelConfig(
            variant="base",
            decoder="mla",
            use_pretrained=True,
            freeze_backbone=True,
        ),
        data=DataLoaderConfig(
            batch_size=8,
            num_workers=4,
            gradient_accumulation=8,
        ),
        runtime=RuntimeConfig(
            epochs=20,
            log_every_n_steps=1,
            patience=10,
            matmul_precision="high",
        ),
        optim=DOFAOptimConfig(
            lr=5.0e-3,
            weight_decay=0.01,
            scheduler_interval="epoch",
        ),
    )