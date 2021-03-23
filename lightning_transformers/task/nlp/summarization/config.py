from dataclasses import dataclass

from lightning_transformers.core.nlp.huggingface.seq2seq.config import HFSeq2SeqTransformerConfig


@dataclass
class SummarizationTransformerConfig(HFSeq2SeqTransformerConfig):
    use_stemmer: bool = True
    rouge_newline_sep: bool = True
