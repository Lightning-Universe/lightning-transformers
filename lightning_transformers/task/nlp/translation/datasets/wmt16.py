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
from typing import Any

from transformers import PreTrainedTokenizerBase

from lightning_transformers.task.nlp.translation.data import TranslationDataModule


class WMT16TranslationDataModule(TranslationDataModule):
    def __init__(self, dataset_name: str = "wmt16", *args, **kwargs) -> None:
        super().__init__(*args, dataset_name=dataset_name, **kwargs)

    @staticmethod
    def convert_to_features(
        examples: Any,
        tokenizer: PreTrainedTokenizerBase,
        padding: str,
        max_source_length: int,
        max_target_length: int,
        src_text_column_name: str,
        tgt_text_column_name: str,
    ):
        inputs = [ex[src_text_column_name] for ex in examples["translation"]]
        targets = [ex[tgt_text_column_name] for ex in examples["translation"]]
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
