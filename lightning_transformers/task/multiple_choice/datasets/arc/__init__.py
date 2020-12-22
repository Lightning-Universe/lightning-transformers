
from lightning_transformers.tasks.multiple_choice.core.data import LitMultipleChoiceDataModule
from .processing import ArcProcessor

class LitArcMultipleChoiceDataModule(LitQuestionAnsweringTransformerDataModule):
    processor: ArcProcessor