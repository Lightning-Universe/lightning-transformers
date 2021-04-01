from dataclasses import dataclass
from typing import Optional

from lightning_transformers.core.nlp import HFTransformerDataConfig


@dataclass
class HFSeq2SeqConfig:
    val_target_max_length: Optional[int] = 128
    num_beams: Optional[int] = 1
    compute_generate_metrics: bool = True


@dataclass
class Seq2SeqDataConfig(HFTransformerDataConfig):
    max_target_length: int = 128
    max_source_length: int = 1024
    padding: str = "longest"
