import os
import torch
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
    default_data_collator,
)
from .data_utils import (
    DataCollatorForMultipleChoice,
    preprocess_function,
)


class LitMultipleChoiceTransformerDataModule(LitTransformerDataModule):

    def process_data(self, dataset: Dataset) -> Dataset:
        from .data_utils import (
            preprocess_function,
        )

        kwargs = self._prepare_preprocess_function_kwargs()
        preprocess_function = partial(preprocess_function, **kwargs)

        dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            load_from_cache_file=not self.overwrite_cache,
        )

        return dataset

    def _prepare_preprocess_function_kwargs(self):
        kwargs = {
            "tokenizer": self.tokenizer,
            "context_name": self.context_name,
            "question_header_name": self.question_header_name,
            "ending_names": self.ending_names,
            "max_seq_length": self.max_seq_length,
            "pad_to_max_length": self.pad_to_max_length,
        }
        return kwargs

    @property
    def data_collator(self):
        return default_data_collator if self.pad_to_max_length \
            else DataCollatorForMultipleChoice(tokenizer=self.tokenizer)

    @property
    def num_classes(self) -> int:
        return len(self.ending_names)

    @property
    def data_model_kwargs(self) -> dict:
        return {
            'num_classes': self.num_classes
        }
