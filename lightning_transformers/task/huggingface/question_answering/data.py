from functools import partial
from typing import Callable, Optional

from datasets import Dataset
from transformers import DataCollatorWithPadding, default_data_collator

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
            "pad_to_max_length": self.cfg.pad_to_max_length,
        }

        prepare_train_features = partial(self.prepare_train_features_function, **kwargs)

        if train:
            dataset["train"] = dataset["train"].map(
                prepare_train_features,
                batched=True,
                num_proc=self.cfg.preprocessing_num_workers,
                remove_columns=dataset["train"].column_names,
                load_from_cache_file=not self.cfg.load_from_cache_file,
            )

        if "test" not in dataset:
            prepare_validation_features = partial(self.prepare_validation_function, **kwargs)
            dataset["validation_original"] = dataset["validation"]
            dataset["validation"] = dataset["validation"].map(
                prepare_validation_features,
                batched=True,
                num_proc=self.cfg.preprocessing_num_workers,
                remove_columns=dataset["validation"].column_names,
                load_from_cache_file=not self.cfg.load_from_cache_file,
            )
        return dataset

    @property
    def collate_fn(self) -> Callable:
        return default_data_collator if self.cfg.pad_to_max_length else DataCollatorWithPadding(self.tokenizer)
