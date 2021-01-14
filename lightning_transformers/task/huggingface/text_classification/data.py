from typing import Any, Dict, List, Optional

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from lightning_transformers.core.huggingface import HFTransformerDataModule


class TextClassificationDataModule(HFTransformerDataModule):
    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        input_feature_fields = [k for k, v in dataset["train"].features.items() if k not in ["label", "idx"]]
        dataset = TextClassificationDataModule.preprocess(
            dataset,
            tokenizer=self.tokenizer,
            input_feature_fields=input_feature_fields,
            padding=self.cfg.padding,
            truncation=self.cfg.truncation,
            max_length=self.cfg.max_length,
        )
        cols_to_keep = [
            x
            for x in ["input_ids", "attention_mask", "token_type_ids", "labels", "idx"]
            if x in dataset["train"].features
        ]
        dataset.set_format("torch", columns=cols_to_keep)
        return dataset

    def prepare_labels(self, dataset: Dataset) -> Any:
        return dataset["train"].features["labels"]

    @property
    def num_classes(self) -> int:
        return self.labels.num_classes

    @property
    def config_data_args(self) -> Dict[str, int]:
        return {"num_labels": self.num_classes}

    @staticmethod
    def convert_to_features(
        example_batch: Any, _, tokenizer: PreTrainedTokenizerBase, input_feature_fields: List[str], **tokenizer_kwargs
    ):
        # Either encode single sentence or sentence pairs
        if len(input_feature_fields) > 1:
            texts_or_text_pairs = list(
                zip(example_batch[input_feature_fields[0]], example_batch[input_feature_fields[1]])
            )
        else:
            texts_or_text_pairs = example_batch[input_feature_fields[0]]
        # Tokenize the text/text pairs
        return tokenizer(texts_or_text_pairs, **tokenizer_kwargs)

    @staticmethod
    def preprocess(ds: Dataset, **fn_kwargs) -> Dataset:
        ds = ds.map(
            TextClassificationDataModule.convert_to_features, batched=True, with_indices=True, fn_kwargs=fn_kwargs
        )
        ds.rename_column_("label", "labels")
        return ds
