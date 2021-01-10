from dataclasses import dataclass
from typing import Optional, Any, Callable, Dict

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


@dataclass
class TransformerDataConfig:
    batch_size: int
    num_workers: int


class TransformerDataModule(pl.LightningDataModule):
    def __init__(self, cfg: TransformerDataConfig):
        super().__init__()
        self.cfg = cfg
        self.ds = None
        self.labels = None

    def setup(self, stage: Optional[str] = None):
        dataset = self.load_dataset()
        dataset = self.split_dataset(dataset)
        dataset = self.process_data(dataset)
        self.labels = self.prepare_labels(dataset)
        self.ds = dataset
        self.load_and_prepare_metrics()

    def load_dataset(self) -> Dict[str, Dataset]:
        raise NotImplementedError

    def split_dataset(self, dataset: Dict[str, Dataset]) -> Dict[str, Dataset]:
        return dataset

    def process_data(self, dataset: Dict[str, Dataset]) -> Dict[str, Dataset]:
        return dataset

    def prepare_labels(self, dataset: Dict[str, Dataset]) -> Optional[Any]:
        return

    def load_and_prepare_metrics(self):
        pass

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
    def config_data_args(self) -> Dict:
        return {}
