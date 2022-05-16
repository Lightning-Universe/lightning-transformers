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
from torchmetrics.text.wer import WordErrorRate
from transformers import Wav2Vec2CTCTokenizer

from lightning_transformers.core.nlp import HFDataModule
from lightning_transformers.task.nlp.speech_recognition.config import (
    SpeechRecognitionConfig, SpeechRecognitionDataConfig
)


class SpeechRecognitionTransformer(HFDataModule):
    """Defines ``LightningModule`` for the SpeechRecognition Task.

    Args:
        *args: :class:`lightning_transformers.core.nlp.seq2seq.Seq2SeqTransformer` arguments.
        downstream_model_type: Downstream HuggingFace AutoModel to load.
            (default ``transformers.AutoModelForSeq2SeqLM``)
        **kwargs: :class:`lightning_transformers.core.nlp.seq2seq.Seq2SeqTransformer` arguments.
    """

    def __init__(
        self,
        *args,
        downstream_model_type: str = "transformers.AutoModelForSeq2SeqLM",
        cfg: SpeechRecognitionConfig = SpeechRecognitionConfig(),
        **kwargs,
    ) -> None:
        super().__init__(downstream_model_type, *args, cfg=cfg, **kwargs)
        self.wer = None

    def on_fit_start(self):
        tokenizer_length = len(self.tokenizer)
        self.model.resize_token_embeddings(tokenizer_length)

    def _step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._step(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._step(batch, batch_idx)
        self.log("test_loss", loss, sync_dist=True)

    def compute_generate_metrics(self, batch, prefix):
        target = self.tokenize_labels(batch["labels"])
        pred = self.generate(batch["input_ids"], batch["attention_mask"])
        # wrap targets in list as score expects a list of potential references
        result = self.wer(pred, target)
        self.log(f"{prefix}_wer_score", result, on_step=False, on_epoch=True, prog_bar=True)

    def configure_metrics(self, stage: str):
        self.wer = WordErrorRate(compute_on_step=self.cfg.compute_on_step)

    def initialize_model_specific_parameters(self):
        super().initialize_model_specific_parameters()
        if isinstance(self.tokenizer, Wav2Vec2CTCTokenizer):
            cfg: SpeechRecognitionDataConfig = self.trainer.datamodule.cfg

    @property
    def hf_pipeline_task(self) -> str:
        return "speech_recognition"
