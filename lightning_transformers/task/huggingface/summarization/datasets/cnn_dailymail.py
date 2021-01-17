from datasets import Dataset, Optional, Tuple

from lightning_transformers.task.huggingface.summarization import SummarizationDataModule


class CNNDailyMailSummarizationDataModule(SummarizationDataModule):
    def source_target_column_names(self, dataset: Dataset, stage: Optional[str] = None) -> Tuple[str, str]:
        return "article", "highlights"
