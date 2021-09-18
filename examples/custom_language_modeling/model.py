import torch

from lightning_transformers.task.nlp.language_modeling import LanguageModelingTransformer


class MyLanguageModelingTransformer(LanguageModelingTransformer):
    def setup(self, stage):
        # Freeze BERT backbone
        for param in self.model.bert.parameters():
            param.requires_grad = False

    def training_step(self, batch, batch_idx):
        # # Add weight noise every training step
        with torch.no_grad():
            for param in self.model.parameters():
                param.add_(torch.randn(param.size()) * 0.1)

        return super().training_step(batch, batch_idx)

    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(parameters, lr=1e-5)
