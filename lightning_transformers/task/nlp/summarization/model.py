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
from typing import Type

import transformers
from torchmetrics.text.rouge import ROUGEScore
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from lightning_transformers.core.seq2seq.model import Seq2SeqTransformer


class SummarizationTransformer(Seq2SeqTransformer):
    """Defines ``LightningModule`` for the Summarization Task.

    Args:
        *args: :class:`lightning_transformers.core.model.TaskTransformer` arguments.
        downstream_model_type: Downstream HuggingFace AutoModel to load.
            (default ``transformers.AutoModelForSeq2SeqLM``)
        use_stemmer: Use Porter stemmer to strip word suffixes to improve matching.
        **kwargs: :class:`lightning_transformers.core.model.TaskTransformer` arguments.
    """

    def __init__(
        self,
        *args,
        downstream_model_type: Type[_BaseAutoModelClass] = transformers.AutoModelForSeq2SeqLM,
        use_stemmer: bool = True,
        **kwargs
    ) -> None:
        super().__init__(downstream_model_type, *args, **kwargs)
        self.rouge = None
        self.use_stemmer = use_stemmer

    def compute_generate_metrics(self, batch, prefix):
        tgt_lns = self.tokenize_labels(batch["labels"])
        pred_lns = self.generate(batch["input_ids"], batch["attention_mask"])
        result = self.rouge(pred_lns, tgt_lns)
        self.log_dict(result, on_step=False, on_epoch=True)

    def configure_metrics(self, stage: str):
        self.rouge = ROUGEScore(use_stemmer=self.use_stemmer)

    @property
    def hf_pipeline_task(self) -> str:
        return "summarization"
