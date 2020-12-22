
from lightning_transformers.task.multiple_choice.core.data import LitMultipleChoiceDataModule
from .processing import SwagProcessor

class LitSwagMultipleChoiceDataModule(LitMultipleChoiceDataModule):

    @property
    def processor(self):
        return SwagProcessor()