from dataclasses import dataclass

from lightning_transformers.core.config import TransformerDataConfig


@dataclass
class TextClassificationDataConfig(TransformerDataConfig):
    ...
