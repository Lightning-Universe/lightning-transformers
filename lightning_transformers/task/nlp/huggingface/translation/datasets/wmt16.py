from typing import Any

from transformers import PreTrainedTokenizerBase

from lightning_transformers.task.nlp.huggingface.translation.data import TranslationDataModule


class WMT16TranslationDataModule(TranslationDataModule):

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
        translations = examples["translation"]  # Extract translations from dict

        def extract_text(lang):
            return [text[lang] for text in translations]

        encoded_results = tokenizer.prepare_seq2seq_batch(
            src_texts=extract_text(src_text_column_name),
            tgt_texts=extract_text(tgt_text_column_name),
            max_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
        )
        return encoded_results
