from typing import Tuple

from lightning_transformers.task.huggingface.summarization import SummarizationDataModule


class CNNDailyMailSummarizationDataModule(SummarizationDataModule):
    @property
    def source_target_column_names(self) -> Tuple[str, str]:
        return "article", "highlights"
