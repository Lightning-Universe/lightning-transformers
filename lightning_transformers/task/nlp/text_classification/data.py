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
from typing import Any, List, Optional

from datasets import ClassLabel, Dataset
from pytorch_lightning.utilities import rank_zero_warn
from transformers import PreTrainedTokenizerBase

from lightning_transformers.core import TransformerDataModule


class TextClassificationDataModule(TransformerDataModule):
    """Defines the ``LightningDataModule`` for Text Classification Datasets."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.labels = None

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        input_feature_fields = [k for k, v in dataset["train"].features.items() if k not in ["label", "idx"]]
        dataset = TextClassificationDataModule.preprocess(
            dataset,
            tokenizer=self.tokenizer,
            input_feature_fields=input_feature_fields,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
        )
        cols_to_keep = [
            x for x in ["input_ids", "attention_mask", "token_type_ids", "labels"] if x in dataset["train"].features
        ]
        if not isinstance(dataset["train"].features["labels"], ClassLabel):
            dataset = dataset.class_encode_column("labels")
        dataset.set_format("torch", columns=cols_to_keep)
        self.labels = dataset["train"].features["labels"]
        return dataset

    @property
    def num_classes(self) -> int:
        if self.labels is None:
            rank_zero_warn("Labels has not been set, calling `setup('fit')`.")
            self.setup("fit")
        return self.labels.num_classes

    @staticmethod
    def convert_to_features(
        example_batch: Any, _, tokenizer: PreTrainedTokenizerBase, input_feature_fields: List[str], **tokenizer_kwargs
    ):
        # Either encode single sentence or sentence pairs
        if len(input_feature_fields) > 1:
            texts_or_text_pairs = list(
                zip(example_batch[input_feature_fields[0]], example_batch[input_feature_fields[1]])
            )
        else:
            texts_or_text_pairs = example_batch[input_feature_fields[0]]
        # Tokenize the text/text pairs
        return tokenizer(texts_or_text_pairs, **tokenizer_kwargs)

    @staticmethod
    def preprocess(ds: Dataset, **fn_kwargs) -> Dataset:
        ds = ds.map(
            # todo: change this to self.convert_to_features for users to override
            TextClassificationDataModule.convert_to_features,
            batched=True,
            with_indices=True,
            fn_kwargs=fn_kwargs,
        )
        ds = ds.rename_column("label", "labels")
        return ds
