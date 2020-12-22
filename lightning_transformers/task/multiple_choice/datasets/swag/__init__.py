
from lightning_transformers.tasks.multiple_choice.core.data import LitMultipleChoiceDataModule
from .processing import SwagProcessor

class LitSwagMultipleChoiceDataModule(LitQuestionAnsweringTransformerDataModule):
    processor: SwagProcessor