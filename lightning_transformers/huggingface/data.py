from dataclasses import dataclass
from typing import Optional

from datasets import Dataset, load_dataset
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from transformers import PreTrainedTokenizerBase

from lightning_transformers.core import TransformerDataModule
from lightning_transformers.core.data import TransformerDataConfig


@dataclass
class HFTransformerDataConfig(TransformerDataConfig):
    dataset_name: Optional[str] = None
    train_val_split: Optional[int] = None
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    padding: str = "max_length"
    truncation: str = "only_first"
    max_length: int = 128
    preprocessing_num_workers: int = 8
    load_from_cache_file: bool = True
    dataset_config_name: Optional[str] = None


class HFTransformerDataModule(TransformerDataModule):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, cfg: HFTransformerDataConfig):
        # TODO: we can save the tokenizer here instead of in the LitModule
        # after https://github.com/PyTorchLightning/pytorch-lightning/pull/3792
        self.tokenizer = tokenizer
        super().__init__(cfg)

    def load_dataset(self) -> Dataset:
        if self.cfg.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            return load_dataset(self.cfg.dataset_name, self.cfg.dataset_config_name)
        data_files = {}
        if self.cfg.train_file is not None:
            data_files["train"] = self.cfg.train_file
        if self.cfg.validation_file is not None:
            data_files["validation"] = self.cfg.validation_file
        if not data_files:
            raise MisconfigurationException(
                "You have not specified a dataset name. A custom train and validation file is required"
            )
        extension = self.cfg.train_file.split(".")[-1]
        return load_dataset(extension, data_files=data_files, field="data")

    def split_dataset(self, dataset: Dataset) -> Dataset:
        if self.cfg.train_val_split is not None:
            split = dataset["train"].train_test_split(self.cfg.train_val_split)
            dataset["train"] = split["train"]
            dataset["validation"] = split["test"]
        return dataset
