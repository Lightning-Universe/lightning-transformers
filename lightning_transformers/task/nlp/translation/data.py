from typing import Tuple

from lightning_transformers.core.nlp.seq2seq import Seq2SeqDataModule
from lightning_transformers.task.nlp.translation.config import TranslationDataConfig


class TranslationDataModule(Seq2SeqDataModule):
    """
    Defines the ``LightningDataModule`` for Translation Datasets.

    Args:
        *args: ``Seq2SeqDataModule`` specific arguments.
        cfg: Contains data specific parameters when processing/loading the dataset
            (Default ``TranslationDataConfig``)
        **kwargs: ``Seq2SeqDataModule`` specific arguments.
    """
    cfg: TranslationDataConfig

    def __init__(self, *args, cfg: TranslationDataConfig = TranslationDataConfig(), **kwargs) -> None:
        super().__init__(*args, cfg=cfg, **kwargs)

    @property
    def source_target_column_names(self) -> Tuple[str, str]:
        return self.cfg.source_language, self.cfg.target_language
