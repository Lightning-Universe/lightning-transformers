# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING, Any, Dict

import torch
from torchmetrics import WordErrorRate

from lightning_transformers.core import TaskTransformer

if TYPE_CHECKING:
    from transformers import Pipeline


class SpeechRecognitionTransformer(TaskTransformer):
    """Defines ``LightningModule`` for the Text Classification Task.

    Args:
        *args: :class:`lightning_transformers.core.nlp.HFTransformer` arguments.
        downstream_model_type: Downstream HuggingFace AutoModel to load.
            (default ``transformers.AutoModelForSequenceClassification``)
        **kwargs: :class:`lightning_transformers.core.nlp.HFTransformer` arguments.
    """

    def __init__(
        self, *args, downstream_model_type: str = "transformers.AutoModelForSequenceClassification", **kwargs
    ) -> None:
        super().__init__(downstream_model_type, *args, **kwargs)
        self.metrics = {}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self.common_step("val", batch)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        if -1 in batch["labels"]:
            batch["labels"] = None
        return self.common_step("test", batch)


    @property
    def hf_pipeline_task(self) -> str:
        return "speech-recognition"

    @property
    def hf_pipeline(self) -> "Pipeline":
        self._hf_pipeline_kwargs["feature_extractor"] = self.tokenizer
        return super().hf_pipeline
