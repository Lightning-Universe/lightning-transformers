from typing import Union

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from lightning_transformers.task.nlp.language_modeling import LanguageModelingDataModule


class MyLanguageModelingDataModule(LanguageModelingDataModule):

    @staticmethod
    def tokenize_function(
        examples,
        tokenizer: Union[Tokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast],
        text_column_name: str = None,
    ):
        examples = examples[text_column_name]
        # append our conditional term to each example
        examples = ['Our conditional term ' + example for example in examples]
        # tokenizes the data in a specific column using the AutoTokenizer, called by `process_data`
        return tokenizer(examples)

    @staticmethod
    def convert_to_features(examples, block_size: int = None):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
