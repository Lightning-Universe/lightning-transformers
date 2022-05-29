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
from typing import Any, Dict, Optional

from datasets import ClassLabel, Dataset
from pytorch_lightning.utilities import rank_zero_warn

from lightning_transformers.core import TransformerDataModule
from lightning_transformers.task.nlp.speech_recognition.config import SpeechRecognitionDataConfig


class SpeechRecognitionDataModule(TransformerDataModule):
    """Defines the ``LightningDataModule`` for Image Classification Datasets."""

    cfg: SpeechRecognitionDataConfig

    def __init__(
        self, feature_extractor, *args, cfg: SpeechRecognitionDataConfig = SpeechRecognitionDataConfig(), **kwargs
    ) -> None:
        super().__init__(tokenizer=feature_extractor, *args, cfg=cfg, **kwargs)
        self.labels = None

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        input_feature_fields = [
            k for k, v in dataset["train"].features.items() if k not in ["label", "sentence", "text", "path"]
        ]


        self.labels = dataset["train"].features["sentence"]
        return dataset
