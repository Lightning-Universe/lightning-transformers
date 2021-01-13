import os
from dataclasses import dataclass
from functools import partial

from datasets import load_metric, Dataset, Union
from tokenizers import Tokenizer
from transformers import (
    default_data_collator,
    DataCollatorWithPadding,
    EvalPrediction, PreTrainedTokenizer, PreTrainedTokenizerFast
)

from lightning_transformers.core.huggingface import HFTransformerDataModule
from lightning_transformers.core.huggingface.config import HFTransformerDataConfig


@dataclass
class QuestionAnsweringTransformerDataConfig(HFTransformerDataConfig):
    max_seq_length: int = 128
    pad_to_max_length: bool = True
    do_train: bool = True
    doc_stride: int = 128
    version_2_with_negative: bool = False
    n_best_size: int = 20
    max_answer_length: int = 30
    null_score_diff_threshold: float = .0
    output_dir: str = './'


class QuestionAnsweringTransformerDataModule(HFTransformerDataModule):

    def __init__(self,
                 tokenizer: Union[Tokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast],
                 cfg: QuestionAnsweringTransformerDataConfig):
        super().__init__(cfg, tokenizer)
        self.cfg = cfg

    def process_data(self, dataset: Dataset) -> Dataset:
        question_column_name, context_column_name, answer_column_name = self.qa_column_names(dataset)

        kwargs = self.prepare_features_kwargs(
            answer_column_name=answer_column_name,
            context_column_name=context_column_name,
            question_column_name=question_column_name
        )

        prepare_train_features = partial(self.prepare_train_features_function, **kwargs)

        if self.cfg.do_train:
            dataset["train"] = dataset["train"].map(
                prepare_train_features,
                batched=True,
                num_proc=self.cfg.preprocessing_num_workers,
                remove_columns=dataset["train"].column_names,
                load_from_cache_file=not self.cfg.load_from_cache_file,
            )

        if "test" not in dataset:
            prepare_validation_features = partial(self.prepare_validation_function, **kwargs)
            dataset['validation_original'] = dataset['validation']
            dataset["validation"] = dataset["validation"].map(
                prepare_validation_features,
                batched=True,
                num_proc=self.cfg.preprocessing_num_workers,
                remove_columns=dataset["validation"].column_names,
                load_from_cache_file=not self.cfg.load_from_cache_file,
            )
        return dataset

    def load_and_prepare_metrics(self):
        self.load_metrics()

        kwargs = self.post_process_kwargs()

        post_process_function = partial(self.post_process_function, **kwargs)

        self.calculate_metrics = partial(self.calculate_metrics, post_process_function=post_process_function)

    @staticmethod
    def prepare_train_features_function():
        pass

    @staticmethod
    def prepare_validation_function():
        pass

    def prepare_features_kwargs(self, answer_column_name, context_column_name, question_column_name):
        return {
            "tokenizer": self.tokenizer,
            "pad_on_right": self.pad_on_right,
            "question_column_name": question_column_name,
            "context_column_name": context_column_name,
            "answer_column_name": answer_column_name,
            "max_length": self.cfg.max_length,
            "doc_stride": self.cfg.doc_stride,
            "pad_to_max_length": self.cfg.pad_to_max_length
        }

    @staticmethod
    def post_process_function():
        pass

    def post_process_kwargs(self):
        return {
            "features": self.ds['validation'],
            "examples": self.ds['validation_original'],
            "version_2_with_negative": self.cfg.version_2_with_negative,
            "n_best_size": self.cfg.n_best_size,
            "max_answer_length": self.cfg.max_answer_length,
            "null_score_diff_threshold": self.cfg.null_score_diff_threshold,
            "output_dir": self.cfg.output_dir,
            "is_world_process_zero": True
        }

    def calculate_metrics(self, predictions, post_process_function=None):
        p = post_process_function(predictions)
        return self.compute_metrics(p)

    def compute_metrics(self, p: EvalPrediction):
        return self.metric.compute(predictions=p.predictions, references=p.label_ids)

    def load_metrics(self):
        current_dir = os.path.sep.join(os.path.join(__file__).split(os.path.sep)[:-1])
        self.metric = load_metric(
            os.path.join(current_dir, "squad_v2_local") if self.cfg.version_2_with_negative else "squad")

    @property
    def pad_on_right(self):
        return self.tokenizer.padding_side == "right"

    def column_names(self, dataset: Dataset):
        if self.cfg.do_train:
            return dataset["train"].column_names
        else:
            return dataset["validation"].column_names

    @property
    def data_collator(self):
        return default_data_collator if self.cfg.pad_to_max_length else DataCollatorWithPadding(self.tokenizer)

    def qa_column_names(self, dataset: Dataset):
        column_names = self.column_names(dataset)
        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        answer_column_name = "answers" if "answers" in column_names else column_names[2]
        return question_column_name, context_column_name, answer_column_name
