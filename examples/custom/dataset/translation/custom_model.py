import torch

from lightning_transformers.task.nlp.translation import TranslationTransformer


class MyTranslationTransformer(TranslationTransformer):

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5)
