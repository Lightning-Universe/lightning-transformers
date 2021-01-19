from functools import partial
from typing import Any, Callable, Optional

from datasets import Dataset
from transformers import DataCollatorWithPadding, default_data_collator, PreTrainedTokenizerBase

from lightning_transformers.core.huggingface import HFTransformerDataModule
from lightning_transformers.task.huggingface.question_answering.config import QuestionAnsweringTransformerDataConfig


class QuestionAnsweringTransformerDataModule(HFTransformerDataModule):
    cfg: QuestionAnsweringTransformerDataConfig

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
            dataset["validation_original"] = dataset["validation"]  # keep an original copy for computing metrics
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
