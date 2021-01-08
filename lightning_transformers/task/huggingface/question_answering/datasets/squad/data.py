from lightning_transformers.task.huggingface.question_answering.core.data import \
    QuestionAnsweringTransformerDataModule
from lightning_transformers.task.huggingface.question_answering.datasets.squad.processing import (
    prepare_train_features,
    prepare_validation_features,
    post_process_function
)


class LitSquadTransformerDataModule(QuestionAnsweringTransformerDataModule):

    @staticmethod
    def prepare_train_features_function(*args, **kwargs):
        return prepare_train_features(*args, **kwargs)

    @staticmethod
    def prepare_validation_function(*args, **kwargs):
        return prepare_validation_features(*args, **kwargs)

    @staticmethod
    def post_process_function(*args, **kwargs):
        return post_process_function(*args, **kwargs)
