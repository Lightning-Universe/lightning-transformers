from dataclasses import dataclass
from typing import Optional, Any

from hydra.utils import get_class

from lightning_transformers.core.base import LitTransformer


@dataclass
class TaskTransformerConfig:
    _target_: Any
    downstream_model_type: str
    backbone: Any
    optim: Any
    scheduler: Any


class TaskTransformer(LitTransformer):
    """
    Base class for task specific transformers, wrapping pre-trained language models for downstream tasks.
    The API is built on top of AutoModel and AutoConfig, provided by HuggingFace.

    see: https://huggingface.co/transformers/model_doc/auto.html
    """

    def __init__(self,
                 cfg: TaskTransformerConfig,
                 config_data_args: Optional[dict]):
        # Resolve the bug in Lightning save_hyperparameters
        cfg.optim.lr = cfg.optim.lr  # todo wat
        self.save_hyperparameters()

        model = get_class(self.hparams.downstream_model_type).from_pretrained(
            self.hparams.cfg.backbone.pretrained_model_name_or_path,
            **self.hparams.config_data_args
        )
        super().__init__(
            model=model,
            optim=self.hparams.cfg.optim,
            scheduler=self.hparams.cfg.scheduler
        )

    def setup(self, stage):
        self.configure_metrics()

    def configure_metrics(self):
        """
        Override to configure metrics for train/validation/test.
        This is called on fit start to have access to the data module,
        and initialize any data specific metrics.
        """
        pass
