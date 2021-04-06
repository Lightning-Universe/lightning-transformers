import torch

from lightning_transformers.task.nlp.language_modeling import LanguageModelingTransformer


class MyLanguageModelingTransformer(LanguageModelingTransformer):

    def on_fit_start(self):
        super().on_fit_start()
        # Freeze backbone
        for param in self.model.transformer.parameters():
            param.requires_grad = False

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)

        # Add weight noise every training step
        with torch.no_grad():
            for param in self.model.parameters():
                param.add_(torch.randn(param.size()) * 0.1)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5)
