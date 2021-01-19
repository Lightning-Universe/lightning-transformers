from typing import Any, Dict

import torch
from datasets import Dataset

from lightning_transformers.task.huggingface.question_answering.data import QuestionAnsweringTransformerDataModule
from lightning_transformers.task.huggingface.question_answering.datasets.squad.processing import (
    post_processing_function,
    prepare_train_features,
    prepare_validation_features,
)


class SquadTransformerDataModule(QuestionAnsweringTransformerDataModule):
    @staticmethod
    def convert_to_train_features(*args, **kwargs):
        return prepare_train_features(*args, **kwargs)

    @staticmethod
    def convert_to_validation_features(*args, **kwargs):
        return prepare_validation_features(*args, **kwargs)

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
            examples=original_validation_dataset,
            features=validation_dataset,
            answer_column_name=self.answer_column_name,
        )
