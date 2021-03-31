import torch

from lightning_transformers.task.nlp.language_modeling import LanguageModelingTransformer


class MyTranslationTransformer(LanguageModelingTransformer):

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5)
