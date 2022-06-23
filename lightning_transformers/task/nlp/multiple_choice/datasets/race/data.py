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
from typing import Any, Dict, Optional

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from lightning_transformers.task.nlp.multiple_choice import MultipleChoiceDataModule


class RaceMultipleChoiceDataModule(MultipleChoiceDataModule):
    def __init__(self, *args, dataset_name: str = "race", dataset_config_name="all", **kwargs):
        super().__init__(*args, dataset_name=dataset_name, dataset_config_name=dataset_config_name, **kwargs)

    @property
    def choices(self) -> list:
        return ["A", "B", "C", "D"]

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        convert_to_features = partial(
            self.convert_to_features,
            tokenizer=self.tokenizer,
            context_name=self.context_name,
            choices=self.choices,
            question_header_name=self.question_header_name,
            answer_column_name=self.answer_column_name,
            options_column_name=self.options_column_name,
            max_length=self.max_length,
            padding=self.padding,
        )

        dataset = dataset.map(
            convert_to_features,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            load_from_cache_file=self.load_from_cache_file,
        )

        cols_to_keep = [
            x
            for x in ["input_ids", "attention_mask", "token_type_ids", "label", "idx"]
            if x in dataset["train"].features
        ]
        dataset.set_format(columns=cols_to_keep)

        return dataset

    @property
    def context_name(self):
        return "article"

    @property
    def question_header_name(self):
        return "question"

    @property
    def answer_column_name(self):
        return "answer"

    @property
    def options_column_name(self):
        return "options"

    @property
    def num_classes(self) -> int:
        return len(self.choices)

    @staticmethod
    def convert_to_features(
        examples: Any,
        tokenizer: PreTrainedTokenizerBase,
        padding: str,
        context_name: str,
        choices: list,
        question_header_name: str,
        answer_column_name: str,
        options_column_name: str,
        max_length: int,
    ) -> Dict:
        num_choices = len(choices)
        first_sentences = [[context] * num_choices for context in examples[context_name]]
        question_headers = examples[question_header_name]
        options = examples[options_column_name]
        second_sentences = [
            [f"{header} {option}" for option in options[i]] for i, header in enumerate(question_headers)
        ]

        # Flatten out
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences, second_sentences, truncation=True, max_length=max_length, padding=padding
        )

        # Un-flatten
        result = {
            k: [v[i : i + num_choices] for i in range(0, len(v), num_choices)] for k, v in tokenized_examples.items()
        }

        label_to_idx = {k: i for i, k in enumerate(choices)}  # convert to label_to_idx
        result["label"] = [label_to_idx[label] for label in examples[answer_column_name]]
        return result
