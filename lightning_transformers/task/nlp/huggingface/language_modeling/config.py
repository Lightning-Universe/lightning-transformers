from lightning_transformers.core.nlp.huggingface.config import HFTransformerDataConfig


class LanguageModelingDataConfig(HFTransformerDataConfig):
    block_size: int = 128
