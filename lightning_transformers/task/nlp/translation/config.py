from dataclasses import dataclass

from lightning_transformers.core.nlp.seq2seq import HFSeq2SeqConfig, Seq2SeqDataConfig


@dataclass
class TranslationConfig(HFSeq2SeqConfig):
    n_gram: int = 4
    smooth: bool = False


@dataclass
class TranslationDataConfig(Seq2SeqDataConfig):
    source_language: str = ""
    target_language: str = ""
