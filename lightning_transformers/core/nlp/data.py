import os
from typing import Any, Dict, Optional, Union

from datasets import Dataset, DatasetDict, load_dataset
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from transformers import PreTrainedTokenizerBase

from lightning_transformers.core.data import TokenizerDataModule
from lightning_transformers.core.nlp.config import HFTransformerDataConfig


class HFDataModule(TokenizerDataModule):
    """Base ``LightningDataModule`` for HuggingFace Datasets. Provides helper functions and boilerplate logic to
    load/process datasets.

    Args:
        tokenizer: ``PreTrainedTokenizerBase`` for tokenizing data.
        cfg: Contains data specific parameters when processing/loading the dataset (Default ``HFTransformerDataConfig``)
    """

    cfg: HFTransformerDataConfig
    tokenizer: PreTrainedTokenizerBase

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, cfg: HFTransformerDataConfig = HFTransformerDataConfig()
    ) -> None:
        super().__init__(tokenizer, cfg=cfg)
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
        if self.cfg.train_file is not None:
            data_files["train"] = self.cfg.train_file
        if self.cfg.validation_file is not None:
            data_files["validation"] = self.cfg.validation_file
        if self.cfg.test_file is not None:
            data_files["test"] = self.cfg.test_file

        data_files = data_files if data_files else None
        if self.cfg.dataset_name is not None:
            # Download and load the Huggingface dataset.
            dataset = load_dataset(
                path=self.cfg.dataset_name,
                name=self.cfg.dataset_config_name,
                cache_dir=self.cfg.cache_dir,
                data_files=data_files,
            )

        # Load straight from data files
        elif self.cfg.datafiles:
            extension = self.cfg.train_file.split(".")[-1]
            dataset = load_dataset(extension, data_files=data_files)

        else:
            raise MisconfigurationException(
                "You have not specified a dataset name nor a custom train and validation file"
            )

        # Use special subset names if provided, and rename them back to standard ones
        for subset in ("train", "validation", "test"):
            config_attr = f"{subset}_subset_name"
            if hasattr(self.cfg, config_attr):
                special_subset_name = getattr(self.cfg, config_attr)
                if special_subset_name not in dataset:
                    raise KeyError(
                        f"Special {subset} subset name {special_subset_name} provided but not found in the dataset"
                    )
                dataset[subset] = dataset.pop(special_subset_name)

        return dataset

    def split_dataset(self, dataset: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
        if self.cfg.train_val_split is not None:
            split = dataset["train"].train_test_split(self.cfg.train_val_split)
            dataset["train"] = split["train"]
            dataset["validation"] = split["test"]
        dataset = self._select_samples(dataset)
        return dataset

    def _select_samples(self, dataset: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
        samples = (
            ("train", self.cfg.limit_train_samples),
            ("validation", self.cfg.limit_val_samples),
            ("test", self.cfg.limit_test_samples),
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
