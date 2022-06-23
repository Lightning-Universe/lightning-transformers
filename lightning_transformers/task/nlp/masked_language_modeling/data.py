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
from typing import Callable, Optional, Union

from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask, PreTrainedTokenizerBase

from lightning_transformers.core import TransformerDataModule


class MaskedLanguageModelingDataModule(TransformerDataModule):
    """Defines ``LightningDataModule`` for Language Modeling Datasets.

    Args:
        *args: ``HFDataModule`` specific arguments.
        mlm_probability: Ratio of tokens to mask for masked language modeling loss
        line_by_line: Whether distinct lines of text in the dataset are to be handled as distinct sequences.
        wwm: Whether or not use whole word masking.
        **kwargs: ``HFDataModule`` specific arguments.
    """

    def __init__(
        self, *args, mlm_probability: float = 0.15, line_by_line: bool = False, wwm: bool = False, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mlm_probability = mlm_probability
        self.line_by_line = line_by_line
        self.wwm = wwm

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        column_names = dataset["train" if stage == "fit" else "validation"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        tokenize_function = partial(
            self.tokenize_function,
            tokenizer=self.tokenizer,
            text_column_name=text_column_name,
            line_by_line=self.line_by_line,
            padding=self.padding,
            max_length=self.max_length,
        )

        dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=self.load_from_cache_file,
        )

        if not self.line_by_line:
            convert_to_features = partial(
                self.convert_to_features,
                max_seq_length=self.max_length,
            )

            dataset = dataset.map(
                convert_to_features,
                batched=True,
                num_proc=self.preprocessing_num_workers,
                load_from_cache_file=self.load_from_cache_file,
            )

        return dataset

    @staticmethod
    def tokenize_function(
        examples,
        tokenizer: Union[PreTrainedTokenizerBase],
        text_column_name: str = None,
        line_by_line: bool = False,
        padding: Union[str, bool] = "max_length",
        max_length: int = 128,
    ):
        if line_by_line:
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
            # efficient when it receives the `special_tokens_mask`.
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

    @staticmethod
    def convert_to_features(examples, max_seq_length: int, **kwargs):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    @property
    def collate_fn(self) -> Callable:
        if self.wwm:
            return DataCollatorForWholeWordMask(self.tokenizer, mlm_probability=self.mlm_probability)
        else:
            return DataCollatorForLanguageModeling(self.tokenizer, mlm_probability=self.mlm_probability)
