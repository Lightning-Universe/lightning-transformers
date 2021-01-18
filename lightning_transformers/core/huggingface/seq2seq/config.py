from dataclasses import dataclass
from typing import Optional


@dataclass
class HFSeq2SeqTransformerConfig:
    val_target_max_length: Optional[int]
    num_beams: Optional[int]
    compute_generate_metrics: bool
