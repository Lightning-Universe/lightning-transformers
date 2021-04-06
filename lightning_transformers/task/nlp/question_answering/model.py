from typing import Any

import torch

from lightning_transformers.core.nlp import HFTransformer


class QuestionAnsweringTransformer(HFTransformer):
    """
    Defines ``LightningModule`` for the Question Answering Task.

    Args:
        *args: :class:`lightning_transformers.core.nlp.HFTransformer` arguments.
        downstream_model_type: Downstream HuggingFace AutoModel to load.
            (default ``transformers.AutoModelForQuestionAnswering``)
        **kwargs: :class:`lightning_transformers.core.nlp.HFTransformer` arguments.
    """

    def __init__(
        self, *args, downstream_model_type: str = 'transformers.AutoModelForQuestionAnswering', **kwargs
    ) -> None:
        super().__init__(downstream_model_type, *args, **kwargs)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log("train_loss", loss)
        return loss

    @property
    def hf_pipeline_task(self) -> str:
        return "question-answering"
