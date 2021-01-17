from typing import Any

import torch
from transformers import MBartTokenizer

from lightning_transformers.core.huggingface.seq2seq.model import HFSeq2SeqTransformer
from lightning_transformers.task.huggingface.translation.config import HFTranslationTransformerConfig, \
    TranslationDataConfig
from lightning_transformers.task.huggingface.translation.metric import BLEUScore


class HFTranslationTransformer(HFSeq2SeqTransformer):
    def __init__(
            self,
            *args,
            cfg: HFTranslationTransformerConfig,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.bleu = None
        self.cfg = cfg

    @property
    def task(self) -> str:
        return "translation"

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        outputs = self.model(**batch)
        loss, logits = outputs[:2]
        self.compute_metrics(logits, batch["labels"], prefix)
        return loss

    def compute_metrics(self, pred, labels, prefix):
        pred = torch.argmax(pred, dim=2)
        pred_lns, tgt_lns = self.decode(pred, labels)
        result = self.bleu(pred_lns, tgt_lns)
        self.log(f'{prefix}_bleu_score', result, on_step=False, on_epoch=True, prog_bar=True)

    def configure_metrics(self, stage: str):
        self.bleu = BLEUScore(
            self.cfg.n_gram,
            self.cfg.smooth
        )

    def initialize_model_specific_parameters(self):
        super().initialize_model_specific_parameters()
        if isinstance(self.tokenizer, MBartTokenizer):
            cfg: TranslationDataConfig = self.trainer.datamodule.cfg
            tgt_lang = cfg.tgt_lang
            # set decoder_start_token_id for MBart
            if self.model.config.decoder_start_token_id is None:
                assert (tgt_lang is not None), "mBart requires --tgt_lang"
                self.model.config.decoder_start_token_id = self.tokenizer.lang_code_to_id[tgt_lang]
