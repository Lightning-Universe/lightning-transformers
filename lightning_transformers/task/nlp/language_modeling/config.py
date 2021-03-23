from lightning_transformers.core.nlp import HFTransformerDataConfig


class LanguageModelingDataConfig(HFTransformerDataConfig):
    block_size: int = 128
