from lightning_transformers.task.nlp.huggingface.question_answering.data import QuestionAnsweringTransformerDataModule
from lightning_transformers.task.nlp.huggingface.question_answering.datasets.squad.processing import (
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
