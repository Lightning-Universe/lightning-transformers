from typing import List

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from lightning_transformers.base import LitTransformer


class LitLanguageModelingTransformer(LitTransformer):
    def __init__(
            self,
            model_name_or_path: str,
            label2id: List[str],
            tokenizer: AutoTokenizer
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            label2id=label2id,
            tokenizer=tokenizer,
            model_type=AutoModelForSequenceClassification
        )


class LitMultipleChoiceTransformer(LitTransformer):
    def __init__(
            self,
            model_name_or_path: str,
            label2id: List[str],
            tokenizer: AutoTokenizer
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            label2id=label2id,
            tokenizer=tokenizer,
            model_type=AutoModelForSequenceClassification
        )


class LitQuestionAnsweringTransformer(LitTransformer):
    def __init__(
            self,
            model_name_or_path: str,
            label2id: List[str],
            tokenizer: AutoTokenizer
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            label2id=label2id,
            tokenizer=tokenizer,
            model_type=AutoModelForSequenceClassification
        )


class LitTextClassificationTransformer(LitTransformer):
    def __init__(
            self,
            model_name_or_path: str,
            label2id: List[str],
            tokenizer: AutoTokenizer
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            label2id=label2id,
            tokenizer=tokenizer,
            model_type=AutoModelForSequenceClassification
        )


class LitTextGenerationTransformer(LitTransformer):
    def __init__(
            self,
            model_name_or_path: str,
            label2id: List[str],
            tokenizer: AutoTokenizer
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            label2id=label2id,
            tokenizer=tokenizer,
            model_type=AutoModelForSequenceClassification
        )


class LitTokenClassificationTransformer(LitTransformer):
    def __init__(
            self,
            model_name_or_path: str,
            label2id: List[str],
            tokenizer: AutoTokenizer
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            label2id=label2id,
            tokenizer=tokenizer,
            model_type=AutoModelForSequenceClassification
        )
