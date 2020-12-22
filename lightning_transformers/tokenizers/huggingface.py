from transformers import AutoTokenizer


def HuggingFaceTokenizer(
        pretrained_model_name_or_path: str,
        use_fast: bool):
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        use_fast=use_fast
    )
