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
from typing import (
    Tuple,
    List,
    Union,
    Optional,
    Dict
)

from datasets import Dataset
from lightning_transformers.core.nlp import HFDataModule
from lightning_transformers.task.nlp.speech_recognition.config import SpeechRecognitionDataConfig

import torch
from transformers import default_data_collator


class SpeechRecognitionDataModule(HFDataModule):
    """Defines the ``LightningDataModule`` for Translation Datasets.

    Args:
        *args: ``SpeechRecognitionDataModule`` specific arguments.
        cfg: Contains data specific parameters when processing/loading the dataset
            (Default ``SpeechRecognitionDataConfig``)
        **kwargs: ``SpeechRecognitionDataModule`` specific arguments.
    """

    cfg: SpeechRecognitionDataConfig

    def __init__(
        self,
        *args,
        cfg: SpeechRecognitionDataConfig = SpeechRecognitionDataConfig(),
        **kwargs
    ) -> None:
        super().__init__(*args, cfg=cfg, **kwargs)
        input_fields="file"
        target_fields="text"
        train_file="data/train.json"
        test_file="data/test.json"

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        train = stage == "fit"
        column_names = dataset["train" if train else "validation"].column_names

        audio_column_name = "audio" if "audio" in column_names else column_names[0]
        sentence_column_name = "sentence" if "sentence" in column_names else column_names[1]
        self.sentence_column_name = sentence_column_name

        kwargs = {
            "tokenizer": self.tokenizer,
            "pad_on_right": self.tokenizer.padding_side == "right",
            "audio_column_name": audio_column_name,
            "sentence_column_name": sentence_column_name,
            "max_length": self.cfg.max_length,
            "padding": self.cfg.padding,
            "sampling_rate": self.cfg.sampling_rate,
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
            kwargs.pop("sentence_column_name")
            prepare_validation_features = partial(
                self.convert_to_validation_features, example_id_strings=self.example_id_strings, **kwargs
            )
            dataset["validation_original"] = dataset["validation"]  # keep an original copy for computing metrics
            dataset["validation"] = dataset["validation"].map(
                prepare_validation_features,
                batched=True,
                num_proc=self.cfg.preprocessing_num_workers,
                remove_columns=column_names,
                # Legacy code `prepare_validation_features` must run to populate `self.example_id_strings`
                # Therefore we cannot load from cache here
                load_from_cache_file=False,
            )
        return dataset

