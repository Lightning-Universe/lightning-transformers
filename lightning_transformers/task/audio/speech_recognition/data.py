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
from typing import Any, Dict, Optional, Union

from datasets import Dataset, Audio
from pytorch_lightning.utilities import rank_zero_warn

from lightning_transformers.core import TransformerDataModule
from lightning_transformers.task.audio.speech_recognition.config import SpeechRecognitionDataConfig
from transformers import PreTrainedTokenizerBase, default_data_collator


class SpeechRecognitionDataModule(TransformerDataModule):
    """Defines the ``LightningDataModule`` for Speech Recognition Datasets."""

    cfg: SpeechRecognitionDataConfig

    def __init__(
        self,
        *args, cfg: SpeechRecognitionDataConfig = SpeechRecognitionDataConfig(),
        **kwargs
    ) -> None:
    
        super().__init__(*args, cfg=cfg, **kwargs)

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:

        # batched output is "un-batched" to ensure mapping is correct
        dataset["input_values"] = processor(dataset["array"], sampling_rate=dataset["sampling_rate"]).input_values[0]
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch
    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        input_feature_fields = [
            k for k, v in dataset["train"].features.items() if k not in ["label", "sentence", "text", "path"]
        ]

        dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
        
        self.labels = dataset["train"].features["sentence"]
        return dataset

    @staticmethod
    def tokenize_function(
        examples,
        tokenizer: Union[PreTrainedTokenizerBase, None],
        text_column_name: str = None,
    ):
        return tokenizer(examples[text_column_name])

    @staticmethod
    def preprocess_function(
        examples,
        tokenizer: Union[AutoProcessor, None],
        text_column_name: str = None,
    ):
        return tokenizer(examples[text_column_name])
