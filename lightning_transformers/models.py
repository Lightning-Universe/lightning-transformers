from typing import List
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)

from lightning_transformers.base import LitTransformer


class LitLanguageModelingTransformer(LitTransformer):
    def __init__(
            self,
            model_name_or_path: str,
            tokenizer: AutoTokenizer
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer=tokenizer,
            model_type=AutoModelForSequenceClassification
        )


class LitMultipleChoiceTransformer(LitTransformer):
    def __init__(
            self,
            model_name_or_path: str,
            tokenizer: AutoTokenizer
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer=tokenizer,
            model_type=AutoModelForSequenceClassification
        )


class LitQuestionAnsweringTransformer(LitTransformer):
    def __init__(
            self,
            model_name_or_path: str,
            tokenizer: AutoTokenizer
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer=tokenizer,
            model_type=AutoModelForQuestionAnswering
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        logits = outputs[0]
        preds = torch.argmax(logits, axis=1)
        metric_dict = self.calculate_metrics(batch, preds)
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)

    def create_metrics(self):
        pass

    def calculate_metrics(self, preds, labels, mode='val'):
        return {}


class LitTextClassificationTransformer(LitTransformer):
    def __init__(
            self,
            model_name_or_path: str,
            tokenizer: AutoTokenizer
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer=tokenizer,
            model_type=AutoModelForSequenceClassification
        )


class LitTextGenerationTransformer(LitTransformer):
    def __init__(
            self,
            model_name_or_path: str,
            tokenizer: AutoTokenizer
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
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
