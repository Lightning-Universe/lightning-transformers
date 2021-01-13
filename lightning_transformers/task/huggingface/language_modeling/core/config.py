from lightning_transformers.core.huggingface.config import HFTransformerDataConfig


class LanguageModelingDataConfig(HFTransformerDataConfig):
    block_size: int = 128
