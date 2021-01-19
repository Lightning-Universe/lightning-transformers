from typing import Any, Dict

import torch

from lightning_transformers.core.huggingface import HFTransformer


class QuestionAnsweringTransformer(HFTransformer):
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        # todo: this needs to be removed here before passing into the model
        # todo: however is needed for metrics computation eventually...
        batch.pop("offset_mapping")
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        outputs = self.model(**batch)
        loss, logits = outputs[:2]
        return loss
