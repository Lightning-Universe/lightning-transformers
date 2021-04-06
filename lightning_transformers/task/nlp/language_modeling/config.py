from dataclasses import dataclass

from lightning_transformers.core.nlp import HFTransformerDataConfig


@dataclass
class LanguageModelingDataConfig(HFTransformerDataConfig):
    block_size: int = 128
