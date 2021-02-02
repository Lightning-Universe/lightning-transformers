from typing import Any, Callable, Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from lightning_transformers.core.config import TransformerDataConfig


class TransformerDataModule(pl.LightningDataModule):

    def __init__(self, cfg: TransformerDataConfig):
        super().__init__()
        self.cfg = cfg
        self.ds = None

    def setup(self, stage: Optional[str] = None):
        dataset = self.load_dataset()
        dataset = self.split_dataset(dataset)
        dataset = self.process_data(dataset, stage=stage)
        self.ds = dataset

    def load_dataset(self) -> Dataset:
        raise NotImplementedError

    def split_dataset(self, dataset: Dataset) -> Dataset:
        return dataset

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        return dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds["train"],
            batch_size=self.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds["validation"],
            batch_size=self.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if "test" in self.ds:
            return DataLoader(
                self.ds["test"],
                batch_size=self.batch_size,
                num_workers=self.cfg.num_workers,
                collate_fn=self.collate_fn,
            )

    @property
    def batch_size(self) -> int:
        return self.cfg.batch_size

    @property
    def collate_fn(self) -> Optional[Callable]:
        return None

    @property
    def model_data_args(self) -> Dict:
        """
        Override to provide the model with additional args.
        This is useful to provide the number of classes/pixels to the model or any other data specific args
        Returns: Dict of args
        """
        return {}


class TransformerTokenizerDataModule(TransformerDataModule):

    def __init__(self, cfg: TransformerDataConfig, tokenizer: Any):
        super().__init__(cfg)
        self.tokenizer = tokenizer
