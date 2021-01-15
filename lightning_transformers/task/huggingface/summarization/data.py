from datasets import Dataset

from lightning_transformers.core.huggingface.seq2seq.data import Seq2SeqDataModule


class SummarizationDataModule(Seq2SeqDataModule):
    def format_dataset_columns(self, dataset: Dataset, src_text_column_name: str, tgt_text_column_name: str):
        dataset.rename_column_("document", src_text_column_name)
        dataset.rename_column_("summary", tgt_text_column_name)

    @property
    def task(self) -> str:
        return "summarization"
