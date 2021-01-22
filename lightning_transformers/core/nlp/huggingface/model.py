from typing import Optional

from hydra.utils import get_class

from lightning_transformers.core.config import OptimizerConfig, SchedulerConfig
from lightning_transformers.core.instantiator import Instantiator
from lightning_transformers.core.model import TaskTransformer
from lightning_transformers.core.nlp.huggingface.config import HFBackboneConfig


class HFTransformer(TaskTransformer):
    """
    Base class for task specific transformers, wrapping pre-trained language models for downstream tasks.
    The API is built on top of AutoModel and AutoConfig, provided by HuggingFace.

    see: https://huggingface.co/transformers/model_doc/auto.html
    """

    def __init__(
        self,
        downstream_model_type: str,
        backbone: HFBackboneConfig,
        optimizer: OptimizerConfig,
        scheduler: SchedulerConfig,
        instantiator: Optional[Instantiator] = None,
        **config_data_args,
    ):
        self.save_hyperparameters()
        model = get_class(downstream_model_type).from_pretrained(
            backbone.pretrained_model_name_or_path, **config_data_args
        )
        super().__init__(model=model, optimizer=optimizer, scheduler=scheduler, instantiator=instantiator)

    @property
    def tokenizer(self):
        return self.trainer.datamodule.tokenizer
