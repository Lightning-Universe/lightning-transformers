from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional

from datasets import Dataset
from transformers import default_data_collator, PreTrainedTokenizerBase

from lightning_transformers.core.huggingface import HFTransformerDataModule
from lightning_transformers.core.huggingface.config import HFTransformerDataConfig


@dataclass
class Seq2SeqDataConfig(HFTransformerDataConfig):
    max_target_length: int = 128
    max_source_length: int = 1024
    padding: str = "longest"


class Seq2SeqDataModule(HFTransformerDataModule):
    cfg: Seq2SeqDataConfig
    src_text_column_name: str = "src_text"
    tgt_text_column_name: str = "tgt_text"

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        self.format_dataset_columns(
            dataset=dataset,
            src_text_column_name=self.src_text_column_name,
            tgt_text_column_name=self.tgt_text_column_name,
        )

        convert_to_features = partial(
            Seq2SeqDataModule.convert_to_features,
            tokenizer=self.tokenizer,
            padding=self.cfg.padding,
            max_source_length=self.cfg.max_source_length,
            max_target_length=self.cfg.max_target_length,
            src_text_column_name=self.src_text_column_name,
            tgt_text_column_name=self.tgt_text_column_name,
        )
        dataset = dataset.map(
            convert_to_features,
            batched=True,
            num_proc=self.cfg.preprocessing_num_workers,
            load_from_cache_file=self.cfg.load_from_cache_file,
        )

        cols_to_keep = [x for x in ["input_ids", "attention_mask", "labels"] if x in dataset["train"].features]
        dataset.set_format(columns=cols_to_keep)
        return dataset

    def format_dataset_columns(self, dataset: Dataset, src_text_column_name: str, tgt_text_column_name: str):
        pass

    def setup_input_fields(self, dataset, stage):
        if stage == "fit":
            column_names = dataset["train"].column_names
            features = dataset["train"].features
        else:
            column_names = dataset["validation"].column_names
            features = dataset["validation"].features
        return features, column_names

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
        encoded_results = tokenizer.prepare_seq2seq_batch(
            src_texts=examples[src_text_column_name],
            tgt_texts=examples[tgt_text_column_name],
            max_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
        )
        return encoded_results

    @property
    def collate_fn(self) -> Optional[Callable]:
        return default_data_collator

    @property
    def task(self):
        raise NotImplementedError
