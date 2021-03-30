# full example of a custom translation model, maybe loading from my own files.
# using HF datasets still to load.

# The docs should explain the basic hook structure as we had before in the HTML file.
# Then show how to override hooks to do different things, explain the structure of what things are doing
# have the example to see the full thing.

from typing import Any

from transformers import PreTrainedTokenizerBase

from lightning_transformers.task.nlp.translation.data import TranslationDataModule


class MyTranslationDataModule(TranslationDataModule):

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
        src_texts = examples[src_text_column_name]
        src_texts = ["Translate from source text: " % src for src in src_texts]
        encoded_results = tokenizer.prepare_seq2seq_batch(
            src_texts=src_texts,
            tgt_texts=examples[tgt_text_column_name],
            max_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
        )
        return encoded_results
