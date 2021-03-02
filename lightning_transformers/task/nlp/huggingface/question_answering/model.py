from typing import Any, Dict, Optional

import torch

from lightning_transformers.core.nlp.huggingface import HFTransformer


class QuestionAnsweringTransformer(HFTransformer):

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log("train_loss", loss)
        return loss

    @property
    def pipeline_task(self) -> Optional[str]:
        return "question-answering"

