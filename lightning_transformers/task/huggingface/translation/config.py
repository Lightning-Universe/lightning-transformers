from dataclasses import dataclass

from lightning_transformers.core.huggingface.seq2seq.data import Seq2SeqDataConfig


@dataclass
class HFTranslationTransformerConfig:
    n_gram: int = 4
    smooth: bool = False


@dataclass
class TranslationDataConfig(Seq2SeqDataConfig):
    src_lang: str = ""
    tgt_lang: str = ""
