import hydra
from omegaconf import DictConfig
from performer_pytorch import AutoregressiveWrapper, PerformerLM

from lightning_transformers.core.model import LitTransformer


class LitPerformerLanguageModelingHFTransformer(LitTransformer):
    def __init__(self,
                 model: DictConfig,
                 optim: DictConfig,
                 scheduler: DictConfig):
        model: PerformerLM = hydra.utils.instantiate(model)
        model = AutoregressiveWrapper(model)

        super().__init__(model, optim, scheduler)

    def step(self, batch, batch_idx):
        loss = self(batch)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)
