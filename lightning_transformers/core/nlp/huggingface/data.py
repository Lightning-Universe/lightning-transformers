import os
from typing import Any, Dict

from datasets import Dataset, load_dataset
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from transformers import PreTrainedTokenizerBase

from lightning_transformers.core.data import TransformerTokenizerDataModule
from lightning_transformers.core.nlp.huggingface.config import HFTransformerDataConfig


class HFTransformerDataModule(TransformerTokenizerDataModule):
    cfg: HFTransformerDataConfig
    tokenizer: PreTrainedTokenizerBase

    def __init__(self, cfg: HFTransformerDataConfig, tokenizer: PreTrainedTokenizerBase):
        super().__init__(cfg, tokenizer)
        os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"  # todo: smarter handling of this env variable

    def load_dataset(self) -> Dataset:
        if self.cfg.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            return load_dataset(
                path=self.cfg.dataset_name, name=self.cfg.dataset_config_name, cache_dir=self.cfg.cache_dir
            )
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
        if getattr(self.cfg, "train_val_split", None) is not None:
            split = dataset["train"].train_test_split(self.cfg.train_val_split)
            dataset["train"] = split["train"]
            dataset["validation"] = split["test"]
        return dataset

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        # Save tokenizer from datamodule for predictions
        checkpoint["tokenizer"] = self.tokenizer

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.tokenizer = checkpoint["tokenizer"]
