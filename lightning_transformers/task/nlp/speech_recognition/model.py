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
import torch
from regex import Match
from torchmetrics.text.wer import WordErrorRate
from torchmetrics.text.cer import CharErrorRate
from torchmetrics.text.mer import MatchErrorRate

from lightning_transformers.core.nlp import HFDataModule
from lightning_transformers.task.nlp.speech_recognition.config import SpeechRecognitionDataConfig, SpeechRecognitionConfig

from typing import (
    Any
)

class SpeechRecognitionTransformer(HFDataModule):
    """Defines ``LightningModule`` for the SpeechRecognition Task.

    Args:
        *args: :class:`lightning_transformers.core.nlp.seq2seq.Seq2SeqTransformer` arguments.
        downstream_model_type: Downstream HuggingFace AutoModel to load.
            (default ``transformers.AutoModelForSpeechSeq2Seq``)
        **kwargs: :class:`lightning_transformers.core.nlp.seq2seq.Seq2SeqTransformer` arguments.
    """

    def __init__(
        self,
        *args,
        downstream_model_type: str = "transformers.AutoModelForSpeechSeq2Seq",
        cfg: SpeechRecognitionConfig = SpeechRecognitionConfig(),
        **kwargs,
    ) -> None:
        super().__init__(downstream_model_type, *args, cfg=cfg, **kwargs)
        self.wer = None

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        
        outputs = self.model(**batch)
        loss, logits = outputs[:2]
        if self.cfg.compute_generate_metrics:
            self.compute_generate_metrics(batch, prefix)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._step(batch, batch_idx, "val")
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._step(batch, batch_idx, "test")
        return loss

    def _step(self, batch, batch_idx, mode):
        outputs = self.model(**batch)
        loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)
        metric_dict = self.compute_metrics(preds, batch["labels"], mode=mode)
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{mode}_loss", loss, prog_bar=True, sync_dist=True)
        return loss



    def configure_metrics(self, stage: str):
        self.wer = WordErrorRate()
        self.cer = CharErrorRate()
        self.mer = MatchErrorRate()
        self.metrics = {"wer": self.wer, "cer": self.cer, "mer": self.mer}

    @property
    def num_classes(self):
        return self.trainer.datamodule.num_classes

    def compute_metrics(self, preds, labels, mode="val"):
        # Remove ignored index (special tokens)
        # Not required by all models. Only required for classification
        return {f"{mode}_{k}": metric(preds, labels) for k, metric in self.metrics.items()}

    @property
    def hf_pipeline_task(self) -> str:
        return "speech_recognition"
