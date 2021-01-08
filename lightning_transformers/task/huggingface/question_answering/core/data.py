import os
from functools import partial

from datasets import load_metric, Dataset
from transformers import (
    default_data_collator,
    DataCollatorWithPadding,
    EvalPrediction
)

from lightning_transformers.core import TransformerDataModule


class QuestionAnsweringTransformerDataModule(TransformerDataModule):

    def __init__(self,
                 max_seq_length: int,
                 pad_to_max_length: int,
                 do_train: bool,
                 doc_stride: int,
                 version_2_with_negative: bool,
                 n_best_size: int,
                 max_answer_length: int,
                 null_score_diff_threshold: float,
                 output_dir: str,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.max_seq_length = max_seq_length
        self.pad_to_max_length = pad_to_max_length
        self.do_train = do_train
        self.doc_stride = doc_stride
        self.version_2_with_negative = version_2_with_negative
        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length
        self.null_score_diff_threshold = null_score_diff_threshold
        self.output_dir = output_dir

    def process_data(self, dataset: Dataset) -> Dataset:
        question_column_name, context_column_name, answer_column_name = self.qa_column_names(dataset)

        kwargs = self.prepare_features_kwargs(
            answer_column_name=answer_column_name,
            context_column_name=context_column_name,
            question_column_name=question_column_name
        )

        prepare_train_features = partial(self.prepare_train_features_function, **kwargs)

        if self.do_train:
            dataset["train"] = dataset["train"].map(
                prepare_train_features,
                batched=True,
                num_proc=self.preprocessing_num_workers,
                remove_columns=dataset["train"].column_names,
                load_from_cache_file=not self.load_from_cache_file,
            )

        if "test" not in dataset:
            prepare_validation_features = partial(self.prepare_validation_function, **kwargs)
            dataset['validation_original'] = dataset['validation']
            dataset["validation"] = dataset["validation"].map(
                prepare_validation_features,
                batched=True,
                num_proc=self.preprocessing_num_workers,
                remove_columns=dataset["validation"].column_names,
                load_from_cache_file=not self.load_from_cache_file,
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
            "max_seq_length": self.max_seq_length,
            "doc_stride": self.doc_stride,
            "pad_to_max_length": self.pad_to_max_length
        }

    @staticmethod
    def post_process_function():
        pass

    def post_process_kwargs(self):
        return {
            "features": self.ds['validation'],
            "examples": self.ds['validation_original'],
            "version_2_with_negative": self.version_2_with_negative,
            "n_best_size": self.n_best_size,
            "max_answer_length": self.max_answer_length,
            "null_score_diff_threshold": self.null_score_diff_threshold,
            "output_dir": self.output_dir,
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
            os.path.join(current_dir, "squad_v2_local") if self.version_2_with_negative else "squad")

    @property
    def pad_on_right(self):
        return self.tokenizer.padding_side == "right"

    def column_names(self, dataset: Dataset):
        if self.do_train:
            return dataset["train"].column_names
        else:
            return dataset["validation"].column_names

    @property
    def data_collator(self):
        return default_data_collator if self.pad_to_max_length else DataCollatorWithPadding(self.tokenizer)

    def qa_column_names(self, dataset: Dataset):
        column_names = self.column_names(dataset)
        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        answer_column_name = "answers" if "answers" in column_names else column_names[2]
        return question_column_name, context_column_name, answer_column_name
