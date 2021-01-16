from datasets import Dataset

from lightning_transformers.task.huggingface.summarization import SummarizationDataModule


class XsumSummarizationDataModule(SummarizationDataModule):
    def format_dataset_columns(self, dataset: Dataset, src_text_column_name: str, tgt_text_column_name: str):
        dataset.rename_column_("document", src_text_column_name)
        dataset.rename_column_("summary", tgt_text_column_name)
