from typing import Any

import torch

from lightning_transformers.core.nlp.huggingface import HFTransformer


class QuestionAnsweringTransformer(HFTransformer):

    def __init__(
        self, *args, downstream_model_type: str = 'transformers.AutoModelForQuestionAnswering', **kwargs
    ) -> None:
        super().__init__(*args, downstream_model_type=downstream_model_type, **kwargs)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log("train_loss", loss)
        return loss

    @property
    def hf_pipeline_task(self) -> str:
        return "question-answering"
