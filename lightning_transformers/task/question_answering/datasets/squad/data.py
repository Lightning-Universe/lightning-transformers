from lightning_transformers.task.question_answering.core.data import LitQuestionAnsweringTransformerDataModule
from lightning_transformers.task.question_answering.datasets.squad import prepare_train_features, \
    prepare_validation_features, post_process_function


class LitSquadTransformerDataModule(LitQuestionAnsweringTransformerDataModule):
    @staticmethod
    def prepare_train_features_function():
        return prepare_train_features

    @staticmethod
    def prepare_validation_function():
        return prepare_validation_features

    @staticmethod
    def post_process_function():
        return post_process_function
