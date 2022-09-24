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
from torchmetrics.text.bleu import BLEUScore
from transformers import MBartTokenizer
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from lightning_transformers.core.seq2seq.model import Seq2SeqTransformer
from lightning_transformers.task.nlp.translation import TranslationDataModule


class TranslationTransformer(Seq2SeqTransformer):
    """Defines ``LightningModule`` for the Translation Task.

    Args:
        *args: :class:`lightning_transformers.core.model.TaskTransformer` arguments.
        downstream_model_type: Downstream HuggingFace AutoModel to load.
            (default ``transformers.AutoModelForSeq2SeqLM``)
        n_gram: Gram value ranged from 1 to 4.
        smooth: Whether or not to apply smoothing.
        **kwargs: :class:`lightning_transformers.core.model.TaskTransformer` arguments.
    """

    def __init__(
        self,
        *args,
        downstream_model_type: Type[_BaseAutoModelClass] = transformers.AutoModelForSeq2SeqLM,
        n_gram: int = 4,
        smooth: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(downstream_model_type, *args, **kwargs)
        self.bleu = None
        self.n_gram = n_gram
        self.smooth = smooth

    def compute_generate_metrics(self, batch, prefix):
        tgt_lns = self.tokenize_labels(batch["labels"])
        pred_lns = self.generate(batch["input_ids"], batch["attention_mask"])
        # wrap targets in list as score expects a list of potential references
        result = self.bleu(preds=pred_lns, target=tgt_lns)
        self.log(f"{prefix}_bleu_score", result, on_step=False, on_epoch=True, prog_bar=True)

    def configure_metrics(self, stage: str):
        self.bleu = BLEUScore(self.n_gram, self.smooth)

    def initialize_model_specific_parameters(self):
        super().initialize_model_specific_parameters()
        if isinstance(self.tokenizer, MBartTokenizer):
            dm: TranslationDataModule = self.trainer.datamodule
            tgt_lang = dm.target_language
            # set decoder_start_token_id for MBart
            if self.model.config.decoder_start_token_id is None:
                assert tgt_lang is not None, "mBart requires --target_language"
                self.model.config.decoder_start_token_id = self.tokenizer.lang_code_to_id[tgt_lang]

    @property
    def hf_pipeline_task(self) -> str:
        return "translation_xx_to_yy"
