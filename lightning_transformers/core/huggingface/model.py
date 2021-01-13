from typing import Dict

from pytorch_lightning import _logger as log

from lightning_transformers.core.config import OptimizerConfig
from lightning_transformers.core.huggingface.config import HFBackboneConfig, HFSchedulerConfig
from lightning_transformers.core.huggingface.instantiator import Instantiator
from lightning_transformers.core.model import TaskTransformer


class HFTransformer(TaskTransformer):
    """
    Base class for task specific transformers, wrapping pre-trained language models for downstream tasks.
    The API is built on top of AutoModel and AutoConfig, provided by HuggingFace.

    see: https://huggingface.co/transformers/model_doc/auto.html
    """

    def __init__(
        self,
        instantiator: Instantiator,
        backbone: HFBackboneConfig,
        optimizer: OptimizerConfig,
        scheduler: HFSchedulerConfig,
    ):
        model = instantiator.backbone(backbone)
        super().__init__(model)
        self.instantiator = instantiator
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler

    def prepare_warmup(self, cfg: HFSchedulerConfig):
        if cfg.num_training_steps < 0:
            # less than 0 specifies to infer number of training steps
            cfg.num_training_steps = self.num_training_steps
            log.info(f"Inferring number of training steps, set to {cfg.num_training_steps}")

        if isinstance(cfg.num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            cfg.num_warmup_steps *= cfg.num_training_steps
            log.info(f"Inferring number of warmup steps from ratio, set to {cfg.num_warmup_steps}")

    def configure_optimizers(self) -> Dict:
        self.optimizer = self.instantiator.optimizer(self.model, self.optimizer_cfg)
        # prepare_warmup needs the datamodule to be available when `self.num_training_steps`
        # is called that is why this is done here and not in the __init__
        self.prepare_warmup(self.scheduler_cfg)
        self.scheduler = self.instantiator.scheduler(self.scheduler_cfg, self.optimizer)
        return super().configure_optimizers()
