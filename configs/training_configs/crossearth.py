"""
Configuration factory for CrossEarth semantic segmentation experiments.

This module defines the complete training configuration used by the CrossEarth
model within the project training pipeline. It combines common configuration
blocks shared across models, such as paths, registry metadata, task settings,
dataloader parameters, and runtime options, with CrossEarth-specific model and
optimization parameters.

CrossEarth is configured here as a multispectral semantic segmentation model
with a DINOv2-style backbone, optional pretrained initialization, trainable
patch embedding, REIN parameters, and a configurable segmentation decoder.

The main entry point is :func:`get_config`, which builds a fully populated
:class:`CrossEarthConfig` instance for a selected dataset.

Typical usage
-------------
>>> from configs.training_configs.crossearth import get_config
>>> cfg = get_config(dataset_name="planetscope")
>>> cfg.model.variant
'dinov2_vitl14_reg'
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
class CrossEarthOptimConfig:
    """
    Optimization configuration for CrossEarth training.

    This block separates the learning rates assigned to the patch embedding,
    REIN parameters, decoder, and backbone. This allows each submodule to be
    optimized with a different update scale, which is useful when combining
    randomly initialized task-specific layers with large pretrained backbones.

    Attributes
    ----------
    lr:
        Base learning rate used as the default optimizer learning rate.
    lr_patch_embed:
        Learning rate assigned to the multispectral patch embedding layer.
    lr_rein:
        Learning rate assigned to REIN-specific parameters.
    lr_decoder:
        Learning rate assigned to the segmentation decoder parameters.
    lr_backbone:
        Learning rate assigned to the backbone parameters.
    weight_decay:
        Weight decay coefficient used by the optimizer.
    warmup_iters:
        Number of scheduler warmup iterations.
    scheduler_interval:
        Scheduler stepping interval, usually ``"step"`` or ``"epoch"``.
    """

    lr: float
    lr_patch_embed: float
    lr_rein: float
    lr_decoder: float
    lr_backbone: float
    weight_decay: float
    warmup_iters: int
    scheduler_interval: str


@dataclass(frozen=True)
class CrossEarthModelConfig:
    """
    Model-specific configuration for the CrossEarth architecture.

    Attributes
    ----------
    variant:
        Backbone variant identifier, for example ``"dinov2_vitl14_reg"``.
    decoder:
        Segmentation decoder type used on top of the backbone features, for
        example ``"mask2former"``.
    decoder_dim:
        Optional decoder embedding dimension. If ``None``, the downstream model
        builder selects its default dimension.
    num_tokens:
        Number of learnable tokens used by CrossEarth-specific adaptation
        modules.
    token_dim:
        Dimensionality of the CrossEarth adaptation tokens.
    dropout:
        Dropout probability applied in the CrossEarth model components where
        supported.
    patch_embed_init:
        Initialization strategy for the multispectral patch embedding layer.
        For example, ``"rgb_mean"`` initializes non-RGB channels from the mean
        of RGB patch embedding weights.
    force_reload:
        Whether to force reloading external backbone assets or cached weights.
    use_pretrained_backbone:
        Whether to initialize the backbone from pretrained weights.
    freeze_backbone:
        Whether to freeze the backbone during training.
    train_patch_embed:
        Whether to keep the multispectral patch embedding layer trainable.
    crossearth_patch_size:
        Patch size configuration used by the CrossEarth preprocessing/training
        pipeline. For single-scale training, this usually contains one value.
    """

    variant: str
    decoder: str
    decoder_dim: int | None
    num_tokens: int
    token_dim: int
    dropout: float
    patch_embed_init: str
    force_reload: bool
    use_pretrained_backbone: bool
    freeze_backbone: bool
    train_patch_embed: bool
    crossearth_patch_size: list[int]


@dataclass(frozen=True)
class CrossEarthConfig:
    """
    Complete experiment configuration for CrossEarth training.

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
    optim:
        CrossEarth-specific optimization configuration.
    model:
        CrossEarth-specific architectural configuration.
    """

    dataset_name: str
    model_name: str
    experiment_name: str

    paths: PathConfig
    registry: RegistryConfig
    task: TaskConfig
    data: DataLoaderConfig
    runtime: RuntimeConfig
    optim: CrossEarthOptimConfig
    model: CrossEarthModelConfig


# ============================================================
# Factory
# ============================================================


def get_config(dataset_name: str = "planetscope") -> CrossEarthConfig:
    """
    Build the default CrossEarth training configuration.

    Parameters
    ----------
    dataset_name:
        Dataset identifier used to derive paths, registry names, and the
        experiment name. Defaults to ``"planetscope"``.

    Returns
    -------
    CrossEarthConfig
        Fully populated immutable configuration object for a CrossEarth
        training run.

    Notes
    -----
    The default configuration uses a ``dinov2_vitl14_reg`` backbone with
    pretrained initialization disabled. Therefore, unless overridden downstream,
    the backbone is configured for training without pretrained weights.

    The task configuration assumes four input channels, matching the
    PlanetScope-style multispectral setup used by the training pipeline.

    The default ``crossearth_patch_size`` is set to ``[504]`` because CrossEarth
    commonly operates on processed 504 x 504 patches derived from 512 x 512 raw
    tiles after architecture-specific spatial constraints are applied.
    """

    model_name = "crossearth"
    experiment_name = f"{model_name}-{dataset_name}"

    return CrossEarthConfig(
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
        data=DataLoaderConfig(
            batch_size=1,
            num_workers=4,
            gradient_accumulation=1,
        ),
        runtime=RuntimeConfig(
            epochs=5,
            log_every_n_steps=1,
            patience=10,
            matmul_precision="high",
        ),
        optim=CrossEarthOptimConfig(
            lr=6.0e-5,
            lr_patch_embed=6.0e-6,
            lr_rein=6.0e-5,
            lr_decoder=6.0e-4,
            lr_backbone=6.0e-6,
            weight_decay=0.05,
            warmup_iters=1500,
            scheduler_interval="step",
        ),
        model=CrossEarthModelConfig(
            variant="dinov2_vitl14_reg",
            decoder="mask2former",
            decoder_dim=None,
            num_tokens=100,
            token_dim=256,
            dropout=0.0,
            patch_embed_init="rgb_mean",
            force_reload=False,
            use_pretrained_backbone=False,
            freeze_backbone=False,
            train_patch_embed=True,
            crossearth_patch_size=[504],
        ),
    )