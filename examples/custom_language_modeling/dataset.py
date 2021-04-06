from functools import partial
from typing import Optional

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from lightning_transformers.core.nlp import HFTransformerDataConfig
from lightning_transformers.task.nlp.language_modeling import LanguageModelingDataModule


class MyLanguageModelingDataModule(LanguageModelingDataModule):

    def __init__(self, cfg: HFTransformerDataConfig, tokenizer: PreTrainedTokenizerBase):
        super().__init__(cfg, tokenizer)
        self.tokenized_condition_term = tokenizer("This is a story: ")

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

        # Pass in our additional condition term when converting to features
        convert_to_features = partial(
            self.convert_to_features,
            block_size=self.effective_block_size,
            tokenized_condition_term=self.tokenized_condition_term
        )

        dataset = dataset.map(
            convert_to_features,
            batched=True,
            num_proc=self.cfg.preprocessing_num_workers,
            load_from_cache_file=self.cfg.load_from_cache_file,
        )

        return dataset

    @staticmethod
    def convert_to_features(examples, block_size: int, **kwargs):
        # Our argument is passed in via kwargs
        tokenized_condition_term = kwargs['tokenized_condition_term']

        # Ensure we consider the conditional term part of the block size
        block_size = block_size - len(tokenized_condition_term)

        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size

        # Add the term to the tokenized blocks of text
        result = {
            k: [tokenized_condition_term + t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
