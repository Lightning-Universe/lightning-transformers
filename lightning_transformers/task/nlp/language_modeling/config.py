from lightning_transformers.core.nlp.huggingface import HFTransformerDataConfig


class LanguageModelingDataConfig(HFTransformerDataConfig):
    block_size: int = 128
