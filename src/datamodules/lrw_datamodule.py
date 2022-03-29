from typing import Optional, Tuple

from numpy import random
from py import process
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from ..utils.lrw_utils import get_preprocessing_pipelines, pad_packed_collate
from .components.lrw_dataset import LRWDataset


class LRWDataModule(LightningDataModule):
    """LightningDataModule for LRW dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/lrw_cropped",
        label_file: str = "data/lrw_cropped/label_sorted.txt",
        visual: bool = True,
        noise_data: str = None,
        annotation_path: Optional[str] = None,
        batch_size: Optional[int] = 16,
        num_workers: Optional[int] = 4,
        pin_memory: Optional[bool] = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = get_preprocessing_pipelines(visual, noise_data)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 500

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # data_dir:str, label_file:str, annotation_dir:str = None, visual: bool = True, partition:str = "train", processing_fn = None
            self.data_train = LRWDataset(
                self.hparams.data_dir,
                self.hparams.label_file,
                self.hparams.annotation_path,
                self.hparams.visual,
                partition="train",
                processing_fn=self.transforms["train"],
            )
            self.data_val = LRWDataset(
                self.hparams.data_dir,
                self.hparams.label_file,
                self.hparams.annotation_path,
                self.hparams.visual,
                partition="val",
                processing_fn=self.transforms["val"],
            )
            self.data_test = LRWDataset(
                self.hparams.data_dir,
                self.hparams.label_file,
                self.hparams.annotation_path,
                self.hparams.visual,
                partition="test",
                processing_fn=self.transforms["val"],
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=pad_packed_collate,
            worker_init_fn=random.seed(1),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=pad_packed_collate,
            worker_init_fn=random.seed(1),
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=pad_packed_collate,
            worker_init_fn=random.seed(1),
        )
