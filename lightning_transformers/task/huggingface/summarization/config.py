from dataclasses import dataclass


@dataclass
class HFSummarizationTransformerConfig:
    use_stemmer: bool = True
    rouge_newline_sep: bool = True
    return_precision_and_recall: bool = True
