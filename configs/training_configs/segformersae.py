"""
Configuration factory for SegFormer-SAE semantic segmentation experiments.

This module defines the complete training configuration used by the
SegFormer-SAE model variant within the project training pipeline. It combines
common configuration blocks shared across models, such as paths, registry
metadata, task settings, dataloader parameters, and runtime options, with
SegFormer-SAE-specific model and optimization parameters.

The main entry point is :func:`get_config`, which builds a fully populated
:class:`SegFormerSAEConfig` instance for a selected dataset.

Typical usage
-------------
>>> from configs.training_configs.segformer_sae import get_config
>>> cfg = get_config(dataset_name="planetscope")
>>> cfg.model.hf_model_name
'nvidia/mit-b2'
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
class SegFormerSAEModelConfig:
    """
    Model-specific configuration for the SegFormer-SAE architecture.

    Attributes
    ----------
    variant:
        SegFormer encoder variant identifier, for example ``"mit-b2"``.
    hf_model_name:
        Hugging Face model identifier used to initialize the SegFormer encoder
        configuration and, optionally, pretrained weights.
    use_pretrained_backbone:
        Whether to initialize the SegFormer encoder backbone from pretrained
        Hugging Face weights. If ``False``, the backbone is trained from scratch.
    use_brd:
        Whether to enable the BRD component in the SegFormer-SAE architecture.
    """

    variant: str
    hf_model_name: str
    use_pretrained_backbone: bool
    use_brd: bool


@dataclass(frozen=True)
class SegFormerSAEOptimConfig:
    """
    Optimization configuration for SegFormer-SAE training.

    This block separates learning rates for the encoder, decoder, and SAE
    components, allowing different optimization dynamics for pretrained or
    randomly initialized submodules.

    Attributes
    ----------
    lr:
        Base learning rate used as the default optimizer learning rate.
    lr_encoder:
        Learning rate assigned to the SegFormer encoder parameters.
    lr_decoder:
        Learning rate assigned to the segmentation decoder parameters.
    lr_sae:
        Learning rate assigned to the SAE-specific parameters.
    weight_decay:
        Weight decay coefficient used by the optimizer.
    loss_alpha:
        Weighting factor used to combine the primary segmentation loss with
        auxiliary SAE-related losses, depending on the Lightning module logic.
    warmup_iters:
        Number of scheduler warmup iterations.
    scheduler_interval:
        Scheduler stepping interval, usually ``"step"`` or ``"epoch"``.
    """

    lr: float
    lr_encoder: float
    lr_decoder: float
    lr_sae: float
    weight_decay: float
    loss_alpha: float
    warmup_iters: int
    scheduler_interval: str


@dataclass(frozen=True)
class SegFormerSAEConfig:
    """
    Complete experiment configuration for SegFormer-SAE training.

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
        ignore index, and class weights.
    data:
        Dataloader configuration, including batch size, workers, and gradient
        accumulation.
    runtime:
        Runtime configuration for training epochs, logging cadence, patience,
        and precision settings.
    model:
        SegFormer-SAE-specific architectural configuration.
    optim:
        SegFormer-SAE-specific optimization configuration.
    """

    dataset_name: str
    model_name: str
    experiment_name: str

    paths: PathConfig
    registry: RegistryConfig
    task: TaskConfig
    data: DataLoaderConfig
    runtime: RuntimeConfig
    model: SegFormerSAEModelConfig
    optim: SegFormerSAEOptimConfig


# ============================================================
# Factory
# ============================================================


def get_config(dataset_name: str = "planetscope") -> SegFormerSAEConfig:
    """
    Build the default SegFormer-SAE training configuration.

    Parameters
    ----------
    dataset_name:
        Dataset identifier used to derive paths, registry names, and the
        experiment name. Defaults to ``"planetscope"``.

    Returns
    -------
    SegFormerSAEConfig
        Fully populated immutable configuration object for a SegFormer-SAE
        training run.

    Notes
    -----
    The default configuration uses the ``nvidia/mit-b2`` SegFormer backbone
    definition but disables pretrained backbone initialization by default via
    ``use_pretrained_backbone=False``. Therefore, unless overridden downstream,
    the model is configured for training from scratch.
    """

    model_name = "segformer_sae"
    experiment_name = f"{model_name}-{dataset_name}"

    return SegFormerSAEConfig(
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
            class_weights=[1.0, 2.5],
        ),
        model=SegFormerSAEModelConfig(
            variant="mit-b2",
            hf_model_name="nvidia/mit-b2",
            use_pretrained_backbone=False,
            use_brd=True,
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
        optim=SegFormerSAEOptimConfig(
            lr=6.0e-5,
            lr_encoder=6.0e-6,
            lr_decoder=6.0e-4,
            lr_sae=6.0e-4,
            weight_decay=0.01,
            loss_alpha=0.7,
            warmup_iters=1500,
            scheduler_interval="step",
        ),
    )