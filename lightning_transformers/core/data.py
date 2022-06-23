# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import pytorch_lightning as pl
from datasets import Dataset, DatasetDict, Version, load_dataset
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from lightning_transformers.core.iterable import IterableDataLoader


class TransformerDataModule(pl.LightningDataModule):
    """Base ``LightningDataModule`` for HuggingFace Datasets. Provides helper functions and boilerplate logic to
    load/process datasets.

    Args:
        tokenizer: ``PreTrainedTokenizerBase`` for tokenizing data.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 32,
        num_workers: int = 0,
        dataset_name: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        revision: Optional[Union[str, Version]] = None,
        train_val_split: Optional[int] = None,
        train_file: Optional[str] = None,
        test_file: Optional[str] = None,
        predict_file: Optional[str] = None,
        validation_file: Optional[str] = None,
        padding: Union[str, bool] = "max_length",
        truncation: str = "only_first",
        max_length: int = 128,
        preprocessing_num_workers: int = 1,
        load_from_cache_file: bool = True,
        cache_dir: Optional[Union[Path, str]] = None,
        limit_train_samples: Optional[int] = None,
        limit_val_samples: Optional[int] = None,
        limit_test_samples: Optional[int] = None,
        train_subset_name: Optional[str] = None,
        validation_subset_name: Optional[str] = None,
        test_subset_name: Optional[str] = None,
        predict_subset_name: Optional[str] = None,
        streaming: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.revision = revision
        self.train_val_split = train_val_split
        self.train_file = train_file
        self.test_file = test_file
        self.predict_file = predict_file
        self.validation_file = validation_file
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.preprocessing_num_workers = preprocessing_num_workers
        self.load_from_cache_file = load_from_cache_file
        self.cache_dir = cache_dir
        self.limit_train_samples = limit_train_samples
        self.limit_val_samples = limit_val_samples
        self.limit_test_samples = limit_test_samples
        self.train_subset_name = train_subset_name
        self.validation_subset_name = validation_subset_name
        self.test_subset_name = test_subset_name
        self.predict_subset_name = predict_subset_name
        self.streaming = streaming
        os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"  # todo: smarter handling of this env variable

    def setup(self, stage: Optional[str] = None):
        dataset = self.load_dataset()
        dataset = self.split_dataset(dataset)
        dataset = self.process_data(dataset, stage=stage)
        self.ds = dataset

    def process_data(
        self, dataset: Union[Dataset, DatasetDict], stage: Optional[str] = None
    ) -> Union[Dataset, DatasetDict]:
        return dataset

    def load_dataset(self) -> Dataset:
        # Allow custom data files when loading the dataset
        data_files = {}
        if self.train_file is not None:
            data_files["train"] = self.train_file
        if self.validation_file is not None:
            data_files["validation"] = self.validation_file
        if self.test_file is not None:
            data_files["test"] = self.test_file

        data_files = data_files if data_files else None
        if self.dataset_name is not None:
            # Download and load the Huggingface dataset.
            dataset = load_dataset(
                path=self.dataset_name,
                name=self.dataset_config_name,
                cache_dir=self.cache_dir,
                data_files=data_files,
                revision=self.revision,
                streaming=self.streaming,
            )

        # Load straight from data files
        elif data_files:
            extension = self.train_file.split(".")[-1]
            dataset = load_dataset(extension, data_files=data_files)

        else:
            raise MisconfigurationException(
                "You have not specified a dataset name nor a custom train and validation file"
            )

        # Use special subset names if provided, and rename them back to standard ones
        for subset in ("train", "validation", "test", "predict"):
            config_attr = f"{subset}_subset_name"
            if getattr(self, config_attr) is not None:
                special_subset_name = getattr(self, config_attr)
                if special_subset_name not in dataset:
                    raise KeyError(
                        f"Special {subset} subset name {special_subset_name} provided but not found in the dataset"
                    )
                dataset[subset] = dataset.pop(special_subset_name)

        return dataset

    def split_dataset(self, dataset: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
        if self.train_val_split is not None:
            split = dataset["train"].train_test_split(self.train_val_split)
            dataset["train"] = split["train"]
            dataset["validation"] = split["test"]
        dataset = self._select_samples(dataset)
        return dataset

    def _select_samples(self, dataset: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
        samples = (
            ("train", self.limit_train_samples),
            ("validation", self.limit_val_samples),
            ("test", self.limit_test_samples),
        )
        for column_name, n_samples in samples:
            if n_samples is not None and column_name in dataset:
                indices = range(min(len(dataset[column_name]), n_samples))
                dataset[column_name] = dataset[column_name].select(indices)
        return dataset

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["tokenizer"] = self.tokenizer

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.tokenizer = checkpoint["tokenizer"]

    def train_dataloader(self) -> DataLoader:
        cls = DataLoader if not self.streaming else IterableDataLoader
        return cls(
            self.ds["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        cls = DataLoader if not self.streaming else IterableDataLoader
        return cls(
            self.ds["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if "test" in self.ds:
            cls = DataLoader if not self.streaming else IterableDataLoader
            return cls(
                self.ds["test"],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )

    def predict_dataloader(self) -> Optional[DataLoader]:
        if "predict" in self.ds:
            cls = DataLoader if not self.streaming else IterableDataLoader
            return cls(
                self.ds["predict"],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )

    @property
    def collate_fn(self) -> Optional[Callable]:
        return None
