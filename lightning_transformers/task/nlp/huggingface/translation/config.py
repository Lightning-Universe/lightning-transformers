from dataclasses import dataclass

from lightning_transformers.core.nlp.huggingface.seq2seq.data import Seq2SeqDataConfig
from lightning_transformers.core.nlp.huggingface.seq2seq.model import HFSeq2SeqTransformerConfig


@dataclass
class HFTranslationTransformerConfig(HFSeq2SeqTransformerConfig):
    n_gram: int = 4
    smooth: bool = False


@dataclass
class TranslationDataConfig(Seq2SeqDataConfig):
    src_lang: str = ""
    tgt_lang: str = ""
