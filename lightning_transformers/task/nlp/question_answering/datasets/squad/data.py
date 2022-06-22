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
from typing import Any, Dict

import torch
from datasets import Dataset

from lightning_transformers.task.nlp.question_answering.data import QuestionAnsweringDataModule
from lightning_transformers.task.nlp.question_answering.datasets.squad.processing import (
    post_processing_function,
    prepare_train_features,
    prepare_validation_features,
)


class SquadDataModule(QuestionAnsweringDataModule):
    def __init__(self, *args, dataset_name: str = "squad", **kwargs):
        super().__init__(*args, dataset_name=dataset_name, **kwargs)

    @staticmethod
    def convert_to_train_features(*args, **kwargs):
        return prepare_train_features(*args, **kwargs)

    @staticmethod
    def convert_to_validation_features(*args, example_id_strings, **kwargs):
        return prepare_validation_features(*args, example_id_strings=example_id_strings, **kwargs)

    def postprocess_func(
        self,
        dataset: Dataset,
        validation_dataset: Dataset,
        original_validation_dataset: Dataset,
        predictions: Dict[int, torch.Tensor],
    ) -> Any:
        return post_processing_function(
            datasets=dataset,
            predictions=predictions,
            answer_column_name=self.answer_column_name,
            features=validation_dataset,
            examples=original_validation_dataset,
            version_2_with_negative=self.version_2_with_negative,
            n_best_size=self.n_best_size,
            max_answer_length=self.max_answer_length,
            null_score_diff_threshold=self.null_score_diff_threshold,
            output_dir=self.output_dir,
        )
