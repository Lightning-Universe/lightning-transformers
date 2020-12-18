from transformers import AutoTokenizer


class HuggingFaceTokenizer:
    def __init__(
            self,
            pretrained_model_name_or_path: str,
            use_fast: bool):
        self.module = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            use_fast=use_fast
        )

    def tokenize(self, *args, **kwargs):
        return self.module.batch_encode_plus(*args, **kwargs)
