from dataclasses import dataclass

from lightning_transformers.core.config import TransformerDataConfig


@dataclass
class SpeechRecognitionDataConfig(TransformerDataConfig):
    ...
