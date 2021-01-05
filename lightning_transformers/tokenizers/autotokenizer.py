from transformers import AutoTokenizer


def TransformersAutoTokenizer(
        pretrained_model_name_or_path: str,
        use_fast: bool):
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        use_fast=use_fast
    )
