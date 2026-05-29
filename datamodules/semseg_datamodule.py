"""
pl_modules/semseg_datamodule.py
--------------------------------
LightningDataModule for streaming semantic segmentation datasets.

This module wraps LitData ``StreamingDataset`` and ``StreamingDataLoader``
in a standard Lightning ``LightningDataModule``, providing a clean interface
for training and validation data loading from pre-optimized chunk directories
produced by ``datasets/opt_dataset.py``.

Requirements
------------
- lightning
- lightning[data]

Usage
-----
    from pl_modules.semseg_datamodule import SemSegStreamingDataModule

    # Default — no model-specific transform
    datamodule = SemSegStreamingDataModule(
        train_dir   = "/path/to/optimized/train",
        val_dir     = "/path/to/optimized/val",
        batch_size  = 8,
        num_workers = 4,
    )

    # CrossEarth — center-crop 512 → 504
    from pl_modules.collates import crossearth_collate
    datamodule = SemSegStreamingDataModule(..., collate_fn=crossearth_collate)

    # DeepLabV3+ / SegFormer-SAE — RGBNIR band selection
    from pl_modules.collates import rgbnir_collate
    datamodule = SemSegStreamingDataModule(..., collate_fn=rgbnir_collate)

    trainer.fit(model=lit_model, datamodule=datamodule)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import lightning.pytorch as pl
from lightning.data import StreamingDataLoader, StreamingDataset


class SemSegStreamingDataModule(pl.LightningDataModule):
    """
    LightningDataModule backed by LitData StreamingDataset.

    Reads pre-optimized chunk directories written by
    ``datasets/opt_dataset.py`` and exposes them as training and
    validation dataloaders.

    Parameters
    ----------
    train_dir : str | Path
        Path to the optimized training chunk directory.
    val_dir : str | Path
        Path to the optimized validation chunk directory.
    batch_size : int
        Number of samples per batch.
    num_workers : int
        Number of dataloader worker processes.
    shuffle_train : bool
        Whether to shuffle the training dataset.
    drop_last : bool
        Whether to drop the last incomplete training batch.
    collate_fn : Callable | None
        Optional collate function for model-specific transforms applied
        at batch-assembly time. Examples:

            crossearth_collate  — center-crop 512 → 504
            rgbnir_collate      — select RGBNIR bands from all-band tensors

        When None, the default LitData collate is used (no transform).
    """

    def __init__(
        self,
        train_dir: str | Path,
        val_dir: str | Path,
        batch_size: int = 8,
        num_workers: int = 4,
        shuffle_train: bool = True,
        drop_last: bool = True,
        collate_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.train_dir = str(train_dir)
        self.val_dir = str(val_dir)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.drop_last = drop_last
        self.collate_fn = collate_fn

        self.train_ds: Optional[StreamingDataset] = None
        self.val_ds: Optional[StreamingDataset] = None

    def setup(self, stage: str | None = None) -> None:
        """
        Instantiate train and validation StreamingDatasets.

        Called by the Lightning trainer before fitting. Datasets are created
        when ``stage`` is ``None``, ``"fit"``, or ``"validate"``.
        """
        if stage in (None, "fit"):
            self.train_ds = StreamingDataset(input_dir=self.train_dir)
            self.val_ds   = StreamingDataset(input_dir=self.val_dir)

        elif stage == "validate":
            self.val_ds = StreamingDataset(input_dir=self.val_dir)

    def train_dataloader(self) -> StreamingDataLoader:
        """
        Build and return the training StreamingDataLoader.

        Returns
        -------
        StreamingDataLoader
            Shuffled dataloader over the training dataset.
        """
        if self.train_ds is None:
            raise RuntimeError(
                "Training dataset is not initialized. "
                "Call setup(stage='fit') before train_dataloader()."
            )

        return StreamingDataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> StreamingDataLoader:
        """
        Build and return the validation StreamingDataLoader.

        Returns
        -------
        StreamingDataLoader
            Non-shuffled dataloader over the validation dataset.
        """
        if self.val_ds is None:
            raise RuntimeError(
                "Validation dataset is not initialized. "
                "Call setup(stage='fit') or setup(stage='validate') "
                "before val_dataloader()."
            )

        return StreamingDataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.collate_fn,
        )