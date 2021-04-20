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
from functools import partial
from typing import Any, Callable, Optional

from datasets import Dataset
from transformers import DataCollatorWithPadding, default_data_collator, PreTrainedTokenizerBase

from lightning_transformers.core.nlp import HFDataModule
from lightning_transformers.task.nlp.question_answering.config import QuestionAnsweringDataConfig


class QuestionAnsweringDataModule(HFDataModule):
    """
    Defines the ``LightningDataModule`` for Question Answering Datasets.

    Args:
        *args: ``HFDataModule`` specific arguments.
        cfg: Contains data specific parameters when processing/loading the dataset
            (Default ``QuestionAnsweringDataConfig``)
        **kwargs: ``HFDataModule`` specific arguments.
    """
    cfg: QuestionAnsweringDataConfig

    def __init__(self, *args, cfg: QuestionAnsweringDataConfig = QuestionAnsweringDataConfig(), **kwargs) -> None:
        super().__init__(*args, cfg=cfg, **kwargs)

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        train = stage == "fit"
        column_names = dataset["train" if train else "validation"].column_names

        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        answer_column_name = "answers" if "answers" in column_names else column_names[2]

        kwargs = {
            "tokenizer": self.tokenizer,
            "pad_on_right": self.tokenizer.padding_side == "right",
            "question_column_name": question_column_name,
            "context_column_name": context_column_name,
            "answer_column_name": answer_column_name,
            "max_length": self.cfg.max_length,
            "doc_stride": self.cfg.doc_stride,
            "padding": self.cfg.padding,
        }

        prepare_train_features = partial(self.convert_to_train_features, **kwargs)

        if train:
            dataset["train"] = dataset["train"].map(
                prepare_train_features,
                batched=True,
                num_proc=self.cfg.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=self.cfg.load_from_cache_file,
            )

        if "test" not in dataset:
            kwargs.pop("answer_column_name")
            prepare_validation_features = partial(self.convert_to_validation_features, **kwargs)
            dataset["validation"] = dataset["validation"].map(
                prepare_validation_features,
                batched=True,
                num_proc=self.cfg.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=self.cfg.load_from_cache_file,
            )
        return dataset

    @property
    def pad_to_max_length(self) -> bool:
        return self.cfg.padding == "max_length"

    @property
    def collate_fn(self) -> Callable:
        return default_data_collator if self.pad_to_max_length else DataCollatorWithPadding(self.tokenizer)

    @staticmethod
    def convert_to_train_features(
        examples: Any,
        tokenizer: PreTrainedTokenizerBase,
        pad_on_right: bool,
        question_column_name: str,
        context_column_name: str,
        answer_column_name: str,
        max_length: int,
        doc_stride: int,
        padding: str,
    ):
        raise NotImplementedError

    @staticmethod
    def convert_to_validation_features(
        examples: Any,
        tokenizer: PreTrainedTokenizerBase,
        pad_on_right: bool,
        question_column_name: str,
        context_column_name: str,
        max_length: int,
        doc_stride: int,
        padding: str,
    ):
        raise NotImplementedError
