from transformers import (
    AutoTokenizer,
)

from lightning_transformers.core.model import LitTransformer


class LitTokenClassificationTransformer(LitTransformer):
    def __init__(
            self,
            pretrained_model_name_or_path: str,
            tokenizer: AutoTokenizer,
            optim_config):
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer=tokenizer,
            optim_config=optim_config,
            model_type=AutoModelForSequenceClassification
        )
