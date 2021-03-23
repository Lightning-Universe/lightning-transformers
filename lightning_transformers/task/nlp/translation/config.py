from dataclasses import dataclass

from lightning_transformers.core.nlp.huggingface import HFSeq2SeqDataConfig, HFSeq2SeqTransformerConfig


@dataclass
class TranslationTransformerConfig(HFSeq2SeqTransformerConfig):
    n_gram: int = 4
    smooth: bool = False


@dataclass
class TranslationDataConfig(HFSeq2SeqDataConfig):
    src_lang: str = ""
    tgt_lang: str = ""
