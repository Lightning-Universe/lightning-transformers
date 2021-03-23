from dataclasses import dataclass

from lightning_transformers.core.nlp.huggingface import HFSeq2SeqTransformerConfig, Seq2SeqDataConfig


@dataclass
class HFTranslationTransformerConfig(HFSeq2SeqTransformerConfig):
    n_gram: int = 4
    smooth: bool = False


@dataclass
class TranslationDataConfig(Seq2SeqDataConfig):
    src_lang: str = ""
    tgt_lang: str = ""
