from functools import partial
from typing import Any, Dict, Optional

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from lightning_transformers.task.huggingface.multiple_choice.data import MultipleChoiceTransformerDataModule


class ArcMultipleChoiceTransformerDataModule(MultipleChoiceTransformerDataModule):
    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        convert_to_features = partial(
            self.convert_to_features,
            tokenizer=self.tokenizer,
            question_header_name=self.question_header_name,
            answer_column_name=self.answer_column_name,
            options_column_name=self.options_column_name,
            max_length=self.cfg.max_length,
            num_classes=self.num_classes,
            padding=self.cfg.padding,
        )

        dataset = dataset.map(
            convert_to_features,
            batched=True,
            num_proc=self.cfg.preprocessing_num_workers,
            load_from_cache_file=self.cfg.load_from_cache_file,
        )

        cols_to_keep = [
            x
            for x in ["input_ids", "attention_mask", "token_type_ids", "label", "idx"]
            if x in dataset["train"].features
        ]
        dataset.set_format(columns=cols_to_keep)

        return dataset

    @property
    def question_header_name(self):
        return "question"

    @property
    def answer_column_name(self):
        return "answerKey"

    @property
    def options_column_name(self):
        return "choices"

    @property
    def num_classes(self) -> int:
        return 4

    @staticmethod
    def convert_to_features(
        examples: Any,
        tokenizer: PreTrainedTokenizerBase,
        padding: str,
        num_classes: int,
        question_header_name: str,
        answer_column_name: str,
        options_column_name: str,
        max_length: int,
    ) -> Dict:
        question_text = examples[question_header_name]
        # create same amounts of copies as number of options
        first_sentences = [[question] * num_classes for question in question_text]

        # extract options
        second_sentences = [option["text"] for option in examples[options_column_name]]

        # Flatten out
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences, second_sentences, truncation=True, max_length=max_length, padding=padding
        )

        # Un-flatten
        result = {
            k: [v[i : i + num_classes] for i in range(0, len(v), num_classes)] for k, v in tokenized_examples.items()
        }

        # convert the given labels per sample into an ID
        # some could be [A, B, C, D] or [1, 2, 3, 4]
        per_batch_labels_idx = [
            {k: v for v, k in enumerate(option["label"])} for option in examples[options_column_name]
        ]
        result["label"] = [per_batch_labels_idx[i][label] for i, label in enumerate(examples[answer_column_name])]
        return result
