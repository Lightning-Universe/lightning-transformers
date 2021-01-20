from functools import partial
from typing import Callable, Optional, Union

from datasets import Dataset
from pytorch_lightning import _logger as log
from tokenizers import Tokenizer
from transformers import default_data_collator, PreTrainedTokenizer, PreTrainedTokenizerFast

from lightning_transformers.core.nlp.huggingface import HFTransformerDataModule
from lightning_transformers.task.nlp.huggingface.language_modeling.config import LanguageModelingDataConfig


class LanguageModelingTransformerDataModule(HFTransformerDataModule):
    cfg: LanguageModelingDataConfig

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        column_names = dataset["train" if stage == "fit" else "validation"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        tokenize_function = partial(self.tokenize_function, tokenizer=self.tokenizer, text_column_name=text_column_name)

        dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.cfg.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=self.cfg.load_from_cache_file,
        )

        group_texts = partial(self.group_texts, block_size=self.effective_block_size)

        dataset = dataset.map(
            group_texts,
            batched=True,
            num_proc=self.cfg.preprocessing_num_workers,
            load_from_cache_file=self.cfg.load_from_cache_file,
        )

        return dataset

    @property
    def effective_block_size(self) -> int:
        if self.cfg.block_size is None:
            block_size = self.tokenizer.model_max_length
            if block_size > 1024:
                log.warn(
                    f"The tokenizer picked seems to have a very large `model_max_length` "
                    f"({self.tokenizer.model_max_length}). "
                    "Picking 1024 instead. You can change that default value by passing --block_size xxx."
                )
            block_size = 1024
        else:
            if self.cfg.block_size > self.tokenizer.model_max_length:
                log.warn(
                    f"The block_size passed ({self.cfg.block_size}) is larger than the maximum length for the model"
                    f"({self.tokenizer.model_max_length}). Using block_size={self.tokenizer.model_max_length}."
                )
            block_size = min(self.cfg.block_size, self.tokenizer.model_max_length)
        return block_size

    @staticmethod
    def tokenize_function(
        examples,
        tokenizer: Union[Tokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast],
        text_column_name: str = None,
    ):
        return tokenizer(examples[text_column_name])

    @staticmethod
    def group_texts(examples, block_size: int = None):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    @property
    def collate_fn(self) -> Callable:
        return default_data_collator
