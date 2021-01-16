from dataclasses import dataclass

from lightning_transformers.core.huggingface.config import HFTransformerDataConfig


@dataclass
class TokenClassificationDataConfig(HFTransformerDataConfig):
    task_name: str = "ner"
    label_all_tokens: bool = False
    pad_to_max_length: bool = False
