
from lightning_transformers.tasks.multiple_choice.core.data import LitMultipleChoiceDataModule
from .processing import RaceProcessor

class LitRaceMultipleChoiceDataModule(LitQuestionAnsweringTransformerDataModule):
    processor: RaceProcessor