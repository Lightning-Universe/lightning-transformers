
from lightning_transformers.tasks.multiple_choice.core.data import LitMultipleChoiceDataModule
from .processing import SynonymProcessor

class LitSynonymMultipleChoiceDataModule(LitQuestionAnsweringTransformerDataModule):
    processor: SynonymProcessor