from argparse import Namespace
from typing import Optional

import hydra
import pytorch_lightning as pl
from datasets import load_dataset
from omegaconf import DictConfig
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader

from lightning_transformers.core.utils import is_overridden


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

    def setup(self, stage: Optional[str] = None):
        self.load_dataset()
        self.split_dataset()
        self.process_data()
        self.prepare_labels()
        self.load_and_prepare_metrics()

    def is_overridden(self, method_name):
        apply_udf = is_overridden(method_name, self, super_object=LitTransformerDataModule)
        if apply_udf:
            udf = getattr(self, method_name)
            udf()
        return apply_udf

    def prepare_labels(self):
        pass

    def _load_and_prepare_metrics(self):
        self.is_overridden("load_and_prepare_metrics")

    @property
    def contains_test(self):
        return 'test' in self.ds

    def process_data(self):
        if self.is_overridden("prepare_pre_processing_functions"):

            if self.args.do_train:
                self.ds["train"] = self.ds["train"].map(
                    self.prepare_train_features,
                    batched=True,
                    num_proc=self.preprocessing_num_workers,
                    remove_columns=self.ds["train"].column_names,
                    load_from_cache_file=not self.load_from_cache_file,
                )

            if not self.contains_test:
                self.ds['validation_orginal'] = self.ds['validation']
                self.ds["validation"] = self.ds["validation"].map(
                    self.prepare_validation_features,
                    batched=True,
                    num_proc=self.preprocessing_num_workers,
                    remove_columns=self.ds["validation"].column_names,
                    load_from_cache_file=not self.load_from_cache_file,
                )

    def load_and_prepare_metrics(self):
        pass

    def load_dataset(self):
        if self.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            self.ds = load_dataset(self.dataset_name, self.dataset_config_name)
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
            self.ds = load_dataset(extension, data_files=data_files, field="data")

    def split_dataset(self):
        if self.train_val_split is not None:
            split = self.ds['train'].train_test_split(self.train_val_split)
            self.ds['train'] = split['train']
            self.ds['validation'] = split['test']

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
