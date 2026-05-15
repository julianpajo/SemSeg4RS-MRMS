
from __future__ import annotations

from typing import Any, Dict


# ---------------------------------------------------------------------
# Training configs
# ---------------------------------------------------------------------

def training_config(
    model_name: str,
    dataset_name: str,
    mode: str,
) -> Dict[str, Any]:
    """
    Build default training hyperparameters.

    These are safe defaults. Tune batch_size, epochs and learning rates
    depending on GPU memory and datasets size.

    Parameters
    ----------
    model_name :
        Target model name.
    dataset_name :
        Dataset/sensor setup name.
    mode :
        Training mode, either "online" or "offline".

    Returns
    -------
    dict
        YAML-ready training configuration.
    """
    cfg = {
        "epochs": 2,
        "batch_size": 1,
        "num_workers": 0,
        "pin_memory": True,
        "drop_last": True,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "grad_clip": 1.0,
        "amp": True,
        "log_every": 1,
        "print_per_class_every": 5,
        "output_dir": "outputs",
        "checkpoint_dir": "outputs/checkpoints",
    }

    if model_name == "crossearth":
        cfg.update(
            {
                "lr_patch_embed": 1.0e-5,
                "lr_rein": 1.0e-4,
                "lr_decoder": 1.0e-3,
                "lr_backbone": 1.0e-5,
            }
        )

    elif model_name == "deeplabv3plus":
        cfg.update(
            {
                "lr": 1.0e-4,
            }
        )

    elif model_name == "segformer_sae":
        cfg.update(
            {
                "lr": 1.0e-4,
            }
        )

    elif model_name == "dofa":
        cfg.update(
            {
                "lr": 1.0e-4,
                "lr_backbone": 1.0e-5,
            }
        )

    # Mixed datasets are usually heavier and safer with smaller batches.
    if dataset_name == "mixed":
        cfg["batch_size"] = min(int(cfg["batch_size"]), 1)

    return cfg


def loss_config(model_name: str) -> Dict[str, Any]:
    """
    Build the loss section.

    build_loss() decides the model-specific loss internally.
    For segformer_sae, it can use WIL if implemented.

    Parameters
    ----------
    model_name :
        Target model name.

    Returns
    -------
    dict
        YAML-ready loss configuration.
    """
    cfg = {
        "ignore_index": 255,
    }

    if model_name == "segformer_sae":
        cfg.update(
            {
                "lambda_dice": 0.5,
                "lambda_focal": 0.5,
                "gamma": 2.0,
            }
        )

    return cfg
