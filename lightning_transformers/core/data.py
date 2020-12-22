from argparse import Namespace
from typing import Optional, Any

import hydra
import pytorch_lightning as pl
from datasets import load_dataset, Dataset
from omegaconf import DictConfig
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader


class LitTransformerDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_name: str,
                 training_config: DictConfig,
                 tokenizer: DictConfig,
                 train_file: Optional[str] = None,
                 validation_file: Optional[str] = None,
                 padding: str = 'max_length',
                 truncation: str = 'only_first',
                 max_length: int = 128,
                 preprocessing_num_workers: int = 8,
                 load_from_cache_file: bool = True,
                 dataset_config_name: Optional[str] = None,
                 train_val_split: Optional[int] = None,
                 **kwargs):
        super().__init__()
        self.args = Namespace(**kwargs, **training_config)
        self.tokenizer = hydra.utils.instantiate(tokenizer)
        self.dataset_name = dataset_name
        self.train_file = train_file
        self.validation_file = validation_file
        self.dataset_config_name = dataset_config_name
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.preprocessing_num_workers = preprocessing_num_workers
        self.load_from_cache_file = load_from_cache_file
        self.train_val_split = train_val_split
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
        if self.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            dataset = load_dataset(self.dataset_name, self.dataset_config_name)
        else:
            if not (self.train_file and self.validation_file):
                raise MisconfigurationException(
                    'You have not specified a dataset name'
                    'and need to specify a custom train file and validation file to the data module.'
                )
            data_files = {}
            if self.train_file is not None:
                data_files["train"] = self.train_file
            if self.validation_file is not None:
                data_files["validation"] = self.validation_file
            extension = self.train_file.split(".")[-1]
            dataset = load_dataset(extension, data_files=data_files, field="data")
        return dataset

    def split_dataset(self, dataset: Dataset) -> Dataset:
        if self.train_val_split is not None:
            split = dataset['train'].train_test_split(self.train_val_split)
            dataset['train'] = split['train']
            dataset['validation'] = split['test']
        return dataset

    def train_dataloader(self):
        return DataLoader(
            self.ds['train'],
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=self.data_collator
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds['validation'],
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=self.data_collator
        )

    def test_dataloader(self):
        dataset = self.ds['test'] if 'test' in self.ds else self.ds['validation']
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=self.data_collator
        )

    @property
    def data_collator(self):
        return None

    @property
    def data_model_kwargs(self):
        return {}
