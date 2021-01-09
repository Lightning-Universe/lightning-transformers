from dataclasses import dataclass
from typing import Optional, Any, Union

import pytorch_lightning as pl
from datasets import load_dataset, Dataset
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


@dataclass
class TransformerDataConfig:
    dataset_name: str
    batch_size: int
    num_workers: int
    train_val_split: Optional[int] = None
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    padding: str = 'max_length'
    truncation: str = 'only_first'
    max_length: int = 128
    preprocessing_num_workers: int = 8
    load_from_cache_file: bool = True
    dataset_config_name: Optional[str] = None


class TransformerDataModule(pl.LightningDataModule):
    def __init__(self,
                 tokenizer: Union[Tokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast],
                 cfg: TransformerDataConfig):
        super().__init__()
        self.tokenizer = tokenizer
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

    def process_data(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError

    def prepare_labels(self, dataset: Dataset) -> Any:
        pass

    def load_and_prepare_metrics(self):
        pass

    def load_dataset(self) -> Dataset:
        if self.cfg.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            dataset = load_dataset(self.cfg.dataset_name, self.cfg.dataset_config_name)
        else:
            if not (self.cfg.train_file and self.cfg.validation_file):
                raise MisconfigurationException(
                    'You have not specified a dataset name'
                    'and need to specify a custom train file and validation file to the data module.'
                )
            data_files = {}
            if self.cfg.train_file is not None:
                data_files["train"] = self.cfg.train_file
            if self.cfg.validation_file is not None:
                data_files["validation"] = self.cfg.validation_file
            extension = self.cfg.train_file.split(".")[-1]
            dataset = load_dataset(extension, data_files=data_files, field="data")
        return dataset

    def split_dataset(self, dataset: Dataset) -> Dataset:
        if self.cfg.train_val_split is not None:
            split = dataset['train'].train_test_split(self.cfg.train_val_split)
            dataset['train'] = split['train']
            dataset['validation'] = split['test']
        return dataset

    def train_dataloader(self):
        return DataLoader(
            self.ds['train'],
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.data_collator
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds['validation'],
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.data_collator
        )

    def test_dataloader(self):
        dataset = self.ds['test'] if 'test' in self.ds else self.ds['validation']
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.data_collator
        )

    @property
    def data_collator(self):
        return None

    @property
    def config_data_args(self):
        return {}
