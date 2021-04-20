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
from typing import Dict

from transformers import default_data_collator

from lightning_transformers.core.nlp import HFDataModule
from lightning_transformers.task.nlp.multiple_choice.utils import DataCollatorForMultipleChoice


class MultipleChoiceDataModule(HFDataModule):
    """
    Defines the ``LightningDataModule`` for Multiple Choice Datasets.
    """

    @property
    def pad_to_max_length(self):
        return self.cfg.padding == "max_length"

    @property
    def collate_fn(self) -> callable:
        return (
            default_data_collator
            if self.pad_to_max_length else DataCollatorForMultipleChoice(tokenizer=self.tokenizer)
        )

    @property
    def num_classes(self) -> int:
        raise NotImplementedError

    @property
    def model_data_kwargs(self) -> Dict[str, int]:
        return {"num_labels": self.num_classes}
