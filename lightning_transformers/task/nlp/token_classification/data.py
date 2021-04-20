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
from typing import Any, Callable, Dict, Optional

from datasets import ClassLabel, Dataset
from transformers import DataCollatorForTokenClassification, PreTrainedTokenizerBase

from lightning_transformers.core.nlp import HFDataModule
from lightning_transformers.task.nlp.token_classification.config import TokenClassificationDataConfig


class TokenClassificationDataModule(HFDataModule):
    """
    Defines the ``LightningDataModule`` for Token Classification Datasets.

    Args:
        *args: ``HFDataModule`` specific arguments.
        cfg: Contains data specific parameters when processing/loading the dataset
            (Default ``TokenClassificationDataConfig``)
        **kwargs: ``HFDataModule`` specific arguments.
    """
    cfg: TokenClassificationDataConfig

    def __init__(self, *args, cfg: TokenClassificationDataConfig = TokenClassificationDataConfig(), **kwargs) -> None:
        super().__init__(*args, cfg=cfg, **kwargs)

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        features, label_column_name, text_column_name = self._setup_input_fields(dataset, stage)

        self._prepare_labels(dataset, features, label_column_name)

        convert_to_features = partial(
            TokenClassificationDataModule.convert_to_features,
            tokenizer=self.tokenizer,
            padding=self.cfg.padding,
            label_all_tokens=self.cfg.label_all_tokens,
            label_to_id=self.label_to_id,
            text_column_name=text_column_name,
            label_column_name=label_column_name,
        )
        dataset = dataset.map(
            convert_to_features,
            batched=True,
            num_proc=self.cfg.preprocessing_num_workers,
            load_from_cache_file=self.cfg.load_from_cache_file,
        )
        cols_to_keep = [
            x for x in ["input_ids", "attention_mask", "token_type_ids", "labels", "idx"]
            if x in dataset["train"].features
        ]
        dataset.set_format(columns=cols_to_keep)
        return dataset

    def _setup_input_fields(self, dataset, stage):
        split = "train" if stage == "fit" else "validation"
        column_names = dataset[split].column_names
        features = dataset[split].features
        text_column_name = "tokens" if "tokens" in column_names else column_names[0]
        label_column_name = (
            f"{self.cfg.task_name}_tags" if f"{self.cfg.task_name}_tags" in column_names else column_names[1]
        )
        return features, label_column_name, text_column_name

    def _prepare_labels(self, dataset, features, label_column_name):
        if isinstance(features[label_column_name].feature, ClassLabel):
            label_list = features[label_column_name].feature.names
            # No need to convert the labels since they are already ints.
            label_to_id = {i: i for i in range(len(label_list))}
        else:
            # Create unique label set from train dataset.
            label_list = sorted(set(label for column in dataset["train"][label_column_name] for label in column))
            label_to_id = {l: i for i, l in enumerate(label_list)}
        self.labels = label_list
        self.label_to_id = label_to_id

    @property
    def num_classes(self) -> int:
        if self.labels is None:
            raise ValueError("Labels has not been set, have you called `datamodule.setup('fit')`?")
        return len(self.labels)

    @property
    def model_data_kwargs(self) -> Dict[str, Any]:
        return {"labels": self.labels}

    @staticmethod
    def convert_to_features(
        examples: Any,
        tokenizer: PreTrainedTokenizerBase,
        padding: bool,
        label_all_tokens: bool,
        label_to_id,
        text_column_name,
        label_column_name,
    ):

        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    @property
    def collate_fn(self) -> Optional[Callable]:
        return DataCollatorForTokenClassification(tokenizer=self.tokenizer)
