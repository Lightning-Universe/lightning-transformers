from dataclasses import dataclass

from lightning_transformers.core.nlp import HFTransformerDataConfig


@dataclass
class TextClassificationDataConfig(HFTransformerDataConfig):
    ...
