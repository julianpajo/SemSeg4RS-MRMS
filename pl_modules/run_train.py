# pl_modules/run_train.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lightning.pytorch as pl
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

from pl_modules.semseg_datamodule import SemSegDataModule
from pl_modules.semseg_module import SemSegLightningModule
from models.factory import apply_train_mode_overrides

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--train-mode",
        default="config",
        choices=["config", "finetune", "scratch"],
        help=(
            "'config' uses the YAML; "
            "'finetune' uses pretrained/frozen where expected; "
            "'scratch' deactivates pretrained e makes the backbone trainable."
        ),
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_train_mode_overrides(cfg, args.train_mode)

    model_type = cfg["model"]["type"]
    train_cfg = cfg["training"]

    run_name = train_cfg.get("run_name", f"{model_type}_lightning")
    output_dir = Path(train_cfg.get("output_dir", "outputs"))
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", output_dir / "checkpoints")) / run_name

    datamodule = SemSegDataModule(cfg)
    module = SemSegLightningModule(cfg)

    trainer = pl.Trainer(
        accelerator="gpu" if cfg.get("device", "cuda") == "cuda" else "cpu",
        devices=[int(args.device)],
        max_epochs=int(train_cfg.get("epochs", 30)),
        precision="16-mixed" if bool(train_cfg.get("amp", True)) else "32-true",
        gradient_clip_val=float(train_cfg.get("grad_clip", 1.0)),
        log_every_n_steps=int(train_cfg.get("log_every", 1)),
        logger=CSVLogger(
            save_dir=output_dir,
            name="lightning_logs",
            version=run_name,
        ),
        callbacks=[
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="best",
                monitor="val/miou",
                mode="max",
                save_top_k=1,
            ),
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="last",
                save_last=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
    )

    trainer.fit(
        model=module,
        datamodule=datamodule,
        ckpt_path=args.resume,
    )


if __name__ == "__main__":
    main()