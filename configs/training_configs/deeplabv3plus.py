"""
Configuration factory for DeepLabV3+ semantic segmentation experiments.

This module defines the complete training configuration used by the
DeepLabV3+ model within the project training pipeline. It combines common
configuration blocks shared across models, such as paths, registry metadata,
task settings, dataloader parameters, and runtime options, with
DeepLabV3+-specific model and optimization parameters.

The main entry point is :func:`get_config`, which builds a fully populated
:class:`DeepLabV3PlusConfig` instance for a selected dataset.

Typical usage
-------------
>>> from configs.training_configs.deeplabv3plus import get_config
>>> cfg = get_config(dataset_name="planetscope")
>>> cfg.model.backbone
'resnet101'
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
class DeepLabV3PlusModelConfig:
    """
    Model-specific configuration for the DeepLabV3+ architecture.

    Attributes
    ----------
    backbone:
        CNN backbone used by DeepLabV3+, for example ``"resnet101"``.
    pretrained_backbone:
        Whether to initialize the backbone from pretrained weights. If
        ``False``, the backbone is trained from scratch.
    """

    backbone: str
    pretrained_backbone: bool


@dataclass(frozen=True)
class DeepLabV3PlusOptimConfig:
    """
    Optimization configuration for DeepLabV3+ training.

    This block separates the learning rates used for the backbone and decoder,
    allowing the encoder to be trained more conservatively than the
    segmentation head when needed.

    Attributes
    ----------
    lr:
        Base learning rate used as the default optimizer learning rate.
    lr_backbone:
        Learning rate assigned to the backbone parameters.
    lr_decoder:
        Learning rate assigned to the decoder and segmentation head parameters.
    weight_decay:
        Weight decay coefficient used by the optimizer.
    warmup_iters:
        Number of scheduler warmup iterations.
    scheduler_interval:
        Scheduler stepping interval, usually ``"step"`` or ``"epoch"``.
    """

    lr: float
    lr_backbone: float
    lr_decoder: float
    weight_decay: float
    warmup_iters: int
    scheduler_interval: str


@dataclass(frozen=True)
class DeepLabV3PlusConfig:
    """
    Complete experiment configuration for DeepLabV3+ training.

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
        Semantic segmentation task configuration, including classes, channels,
        ignore index, and optional class weights.
    data:
        Dataloader configuration, including batch size, workers, and gradient
        accumulation.
    runtime:
        Runtime configuration for training epochs, logging cadence, patience,
        and precision settings.
    model:
        DeepLabV3+-specific architectural configuration.
    optim:
        DeepLabV3+-specific optimization configuration.
    """

    dataset_name: str
    model_name: str
    experiment_name: str

    paths: PathConfig
    registry: RegistryConfig
    task: TaskConfig
    data: DataLoaderConfig
    runtime: RuntimeConfig
    model: DeepLabV3PlusModelConfig
    optim: DeepLabV3PlusOptimConfig


# ============================================================
# Factory
# ============================================================


def get_config(dataset_name: str = "planetscope") -> DeepLabV3PlusConfig:
    """
    Build the default DeepLabV3+ training configuration.

    Parameters
    ----------
    dataset_name:
        Dataset identifier used to derive paths, registry names, and the
        experiment name. Defaults to ``"planetscope"``.

    Returns
    -------
    DeepLabV3PlusConfig
        Fully populated immutable configuration object for a DeepLabV3+
        training run.

    Notes
    -----
    The default configuration uses a ``resnet101`` backbone with pretrained
    initialization disabled. Therefore, unless overridden downstream, the model
    is configured for training from scratch.

    The task configuration assumes four input channels, which matches the
    PlanetScope-style multispectral setup used by the training pipeline.
    """

    model_name = "deeplabv3plus"
    experiment_name = f"{model_name}-{dataset_name}"

    return DeepLabV3PlusConfig(
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
            in_channels=4,
        ),
        model=DeepLabV3PlusModelConfig(
            backbone="resnet101",
            pretrained_backbone=False,
        ),
        data=DataLoaderConfig(
            batch_size=1,
            num_workers=4,
            gradient_accumulation=1,
        ),
        runtime=RuntimeConfig(
            epochs=10,
            log_every_n_steps=1,
            patience=10,
            matmul_precision="high",
        ),
        optim=DeepLabV3PlusOptimConfig(
            lr=1.0e-4,
            lr_backbone=1.0e-5,
            lr_decoder=1.0e-3,
            weight_decay=1.0e-4,
            warmup_iters=1500,
            scheduler_interval="step",
        ),
    )