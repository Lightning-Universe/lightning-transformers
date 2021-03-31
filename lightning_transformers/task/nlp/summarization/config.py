from dataclasses import dataclass

from lightning_transformers.core.nlp.huggingface.seq2seq.config import HFSeq2SeqConfig


@dataclass
class SummarizationConfig(HFSeq2SeqConfig):
    use_stemmer: bool = True
    rouge_newline_sep: bool = True
