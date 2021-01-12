from dataclasses import dataclass
from typing import Dict

from pytorch_lightning import _logger as log

from lightning_transformers.core.model import TaskTransformer
from lightning_transformers.core.huggingface.instantiator import Instantiator


@dataclass
class HFBackboneConfig:
    downstream_model_type: str
    pretrained_model_name_or_path: str


@dataclass
class HFOptimizerConfig:
    weight_decay: float


@dataclass
class HFSchedulerConfig:
    num_training_steps: int
    num_warmup_steps: float


@dataclass
class HFTransformerConfig:
    optimizer: HFOptimizerConfig
    scheduler: HFSchedulerConfig


class HFTransformer(TaskTransformer):
    """
    Base class for task specific transformers, wrapping pre-trained language models for downstream tasks.
    The API is built on top of AutoModel and AutoConfig, provided by HuggingFace.

    see: https://huggingface.co/transformers/model_doc/auto.html
    """

    def __init__(self, instantiator: Instantiator, module: HFTransformerConfig, backbone: HFBackboneConfig):
        model = instantiator.backbone(backbone)
        optimizer = instantiator.optimizer(model, module.optimizer)
        scheduler = instantiator.scheduler(module.scheduler, optimizer)
        super().__init__(model, optimizer, scheduler)
        self._num_training_steps = module.scheduler.num_training_steps
        self._num_warmup_steps = module.scheduler.num_warmup_steps

    def configure_optimizers(self) -> Dict:
        """Prepare optimizer and scheduler (linear warmup and decay)"""
        # TODO: where are these used?
        # TODO: should this be in core instead?
        if self._num_training_steps < 0:
            # less than 0 specifies to infer number of training steps
            self._num_training_steps = self.num_training_steps
            log.info(f"Inferring number of training steps, set to {self._num_training_steps}")

        if isinstance(self._num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            self._num_warmup_steps *= self._num_training_steps
            log.info(f"Inferring number of warmup steps from ratio, set to {self._num_warmup_steps}")

        return super().configure_optimizers()
