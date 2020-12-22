from lightning_transformers.task.question_answering.core.data import LitQuestionAnsweringTransformerDataModule
from lightning_transformers.task.question_answering.datasets.squad.processing import (
    prepare_train_features,
    prepare_validation_features, 
    post_process_function
)

class LitSquadTransformerDataModule(LitQuestionAnsweringTransformerDataModule):
    
    @staticmethod
    def prepare_train_features_function(*args, **kwargs):
        return prepare_train_features(*args, **kwargs)

    @staticmethod
    def prepare_validation_function(*args, **kwargs):
        return prepare_validation_features(*args, **kwargs)

    @staticmethod
    def post_process_function(*args, **kwargs):
        return post_process_function(*args, **kwargs)
