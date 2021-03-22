from transformers import MBartTokenizer

from lightning_transformers.core.nlp.huggingface.seq2seq.model import HFSeq2SeqTransformer
from lightning_transformers.task.nlp.huggingface.translation.config import (
    HFTranslationTransformerConfig,
    TranslationDataConfig,
)
from lightning_transformers.task.nlp.huggingface.translation.metric import BLEUScore


class HFTranslationTransformer(HFSeq2SeqTransformer):
    cfg: HFTranslationTransformerConfig

    def __init__(
        self,
        downstream_model_type='transformers.AutoModelForSeq2SeqLM',
        cfg: HFTranslationTransformerConfig = HFTranslationTransformerConfig(),
        *args,
        **kwargs
    ):
        super().__init__(*args, downstream_model_type=downstream_model_type, cfg=cfg, **kwargs)
        self.bleu = None

    def compute_generate_metrics(self, batch, prefix):
        tgt_lns = self.tokenize_labels(batch["labels"])
        pred_lns = self.generate(batch["input_ids"], batch["attention_mask"])
        # wrap targets in list as score expects a list of potential references
        tgt_lns = [[reference] for reference in tgt_lns]
        result = self.bleu(pred_lns, tgt_lns)
        self.log(f"{prefix}_bleu_score", result, on_step=False, on_epoch=True, prog_bar=True)

    def configure_metrics(self, stage: str):
        self.bleu = BLEUScore(self.cfg.n_gram, self.cfg.smooth)

    def initialize_model_specific_parameters(self):
        super().initialize_model_specific_parameters()
        if isinstance(self.tokenizer, MBartTokenizer):
            cfg: TranslationDataConfig = self.trainer.datamodule.cfg
            tgt_lang = cfg.tgt_lang
            # set decoder_start_token_id for MBart
            if self.model.config.decoder_start_token_id is None:
                assert tgt_lang is not None, "mBart requires --tgt_lang"
                self.model.config.decoder_start_token_id = self.tokenizer.lang_code_to_id[tgt_lang]

    @property
    def hf_pipeline_task(self) -> str:
        return "translation_xx_to_yy"
