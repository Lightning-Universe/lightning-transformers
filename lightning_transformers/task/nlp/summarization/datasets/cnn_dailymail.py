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
from typing import Tuple

from lightning_transformers.task.nlp.summarization import SummarizationDataModule


class CNNDailyMailSummarizationDataModule(SummarizationDataModule):
    def __init__(self, *args, dataset_name: str = "cnn_dailymail", **kwargs):
        super().__init__(*args, dataset_name=dataset_name, **kwargs)

    @property
    def source_target_column_names(self) -> Tuple[str, str]:
        return "article", "highlights"
