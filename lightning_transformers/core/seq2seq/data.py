from functools import partial
from typing import Any, Callable, Optional, Tuple

from datasets import Dataset
from transformers import PreTrainedTokenizerBase, default_data_collator

from lightning_transformers.core import TransformerDataModule


class Seq2SeqDataModule(TransformerDataModule):
    """Defines the ``LightningDataModule`` for Seq2Seq Datasets, such as Summarization and Translation.

    Args:
        *args: ``HFDataModule`` specific arguments.
        **kwargs: ``HFDataModule`` specific arguments.
    """

    def __init__(
        self, *args, max_target_length: int = 128, max_source_length: int = 1024, padding: str = "longest", **kwargs
    ) -> None:
        super().__init__(*args, padding=padding, **kwargs)
        self.max_target_length = max_target_length
        self.max_source_length = max_source_length

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        src_text_column_name, tgt_text_column_name = self.source_target_column_names

        convert_to_features = partial(
            self.convert_to_features,
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
            src_text_column_name=src_text_column_name,
            tgt_text_column_name=tgt_text_column_name,
        )
        dataset = dataset.map(
            convert_to_features,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            load_from_cache_file=self.load_from_cache_file,
        )

        cols_to_keep = [x for x in ["input_ids", "attention_mask", "labels"] if x in dataset["train"].features]
        dataset.set_format(columns=cols_to_keep)
        return dataset

    @property
    def source_target_column_names(self) -> Tuple[str, str]:
        return "source", "target"

    @staticmethod
    def convert_to_features(
        examples: Any,
        tokenizer: PreTrainedTokenizerBase,
        padding: str,
        max_source_length: int,
        max_target_length: int,
        src_text_column_name: str,
        tgt_text_column_name: str,
    ):
        inputs = examples[src_text_column_name]
        targets = examples[tgt_text_column_name]
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    @property
    def collate_fn(self) -> Callable:
        return default_data_collator
