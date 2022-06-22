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

from datasets import Dataset, IterableDataset
from pytorch_lightning import _logger as log
from transformers import PreTrainedTokenizerBase, default_data_collator

from lightning_transformers.core import TransformerDataModule


class LanguageModelingDataModule(TransformerDataModule):
    """Defines ``LightningDataModule`` for Language Modeling Datasets.

    Args:
        *args: ``HFDataModule`` specific arguments.
        **kwargs: ``HFDataModule`` specific arguments.
    """

    def __init__(self, *args, block_size: int = 128, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block_size = block_size

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        dataset_split = dataset["train" if stage == "fit" else "validation"]
        streaming_dataset = isinstance(dataset_split, IterableDataset)
        if streaming_dataset:
            # assume we just have a single text column name
            column_names = ["text"]
            text_column_name = "text"
        else:
            column_names = dataset_split.column_names
            text_column_name = "text" if "text" in column_names else column_names[0]

        tokenize_function = partial(self.tokenize_function, tokenizer=self.tokenizer, text_column_name=text_column_name)
        convert_to_features = partial(self.convert_to_features, block_size=self.effective_block_size)

        if streaming_dataset:
            dataset = dataset.map(tokenize_function, batched=True, remove_columns=column_names)
            return dataset.map(convert_to_features, batched=True)

        dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=self.load_from_cache_file,
        )

        return dataset.map(
            convert_to_features,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            load_from_cache_file=self.load_from_cache_file,
        )

    @property
    def effective_block_size(self) -> int:
        if self.block_size is None:
            block_size = self.tokenizer.model_max_length
            if block_size > 1024:
                log.warn(
                    f"The tokenizer picked seems to have a very large `model_max_length` "
                    f"({self.tokenizer.model_max_length}). "
                    "Picking 1024 instead. You can change that default value by passing block_size=x to the DataModule."
                )
            block_size = 1024
        else:
            if self.block_size > self.tokenizer.model_max_length:
                log.warn(
                    f"The block_size passed ({self.block_size}) is larger than the maximum length for the model"
                    f"({self.tokenizer.model_max_length}). Using block_size={self.tokenizer.model_max_length}."
                )
            block_size = min(self.block_size, self.tokenizer.model_max_length)
        return block_size

    @staticmethod
    def tokenize_function(
        examples,
        tokenizer: Union[PreTrainedTokenizerBase],
        text_column_name: str = None,
    ):
        return tokenizer(examples[text_column_name])

    @staticmethod
    def convert_to_features(examples, block_size: int, **kwargs):
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

    @property
    def collate_fn(self) -> Callable:
        return default_data_collator
