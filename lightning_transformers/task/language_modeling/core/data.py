import os
import torch
from functools import partial
from enum import Enum
from typing import List, Optional
from functools import partial
from dataclasses import dataclass
from datasets import load_metric, Dataset
from transformers import PreTrainedTokenizer
from lightning_transformers.core import LitTransformerDataModule
from filelock import FileLock
import tqdm
from pytorch_lightning import _logger as log
from transformers import (
    AutoTokenizer,
    default_data_collator,
)


class LitLanguageModelingTransformerDataModule(LitTransformerDataModule):

    def process_data(self, dataset: Dataset) -> Dataset:

        if self.do_train:
            column_names = dataset["train"].column_names
        else:
            column_names = dataset["validation"].column_names
        
        text_column_name = "text" if "text" in column_names else column_names[0]

        tokenize_function = partial(
            self.tokenize_function, 
            tokenizer=self.tokenizer,
            text_column_name=text_column_name
        )

        dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not self.overwrite_cache,
        )

        group_texts = partial(self.group_texts, block_size=self.effective_block_size)

        dataset = dataset.map(
            group_texts,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            load_from_cache_file=not self.overwrite_cache,
        )

        return dataset

    @property
    def effective_block_size(self):
        if self.block_size is None:
            block_size = self.tokenizer.model_max_length
            if block_size > 1024:
                logger.warn(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({self.tokenizer.model_max_length}). "
                    "Picking 1024 instead. You can change that default value by passing --block_size xxx."
                )
            block_size = 1024
        else:
            if self.block_size > self.tokenizer.model_max_length:
                logger.warn(
                    f"The block_size passed ({self.block_size}) is larger than the maximum length for the model"
                    f"({self.tokenizer.model_max_length}). Using block_size={self.tokenizer.model_max_length}."
                )
            block_size = min(self.block_size, self.tokenizer.model_max_length)
        return block_size

    @staticmethod
    def tokenize_function(examples, 
                          tokenizer: AutoTokenizer = None,
                          text_column_name: str = None):
        return tokenizer(examples[text_column_name])

    @staticmethod
    def group_texts(examples, 
                    block_size: int = None):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result