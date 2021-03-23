from typing import Tuple

from lightning_transformers.task.nlp.summarization import SummarizationDataModule


class XsumSummarizationDataModule(SummarizationDataModule):

    @property
    def source_target_column_names(self) -> Tuple[str, str]:
        return "document", "summary"
