from typing import Any, Dict

import hydra
import torch

from lightning_transformers.core import TaskTransformer


class HydraTaskTransformer(TaskTransformer):
    """
    Base class for using Hydra instantiation to build task transformers.
    Offers helper functions to setup optimizer via config objects.
    """

    def __init__(self, optimizer_cfg: Any, scheduler_cfg: Any):
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

    def configure_optimizers(self) -> Dict:
        self.optimizer = self.optimizer(self.optimizer_cfg, self.model)
        # prepare_warmup needs the datamodule to be available when `self.num_training_steps`
        # is called that is why this is done here and not in the __init__
        self.prepare_warmup(self.scheduler_cfg)
        self.scheduler = self.scheduler(self.scheduler_cfg, self.optimizer)
        return super().configure_optimizers()

    def optimizer(self, cfg: Any, model: torch.nn.Module) -> torch.optim.Optimizer:
        no_decay = ["bias", "LayerNorm.weight"]
        grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return hydra.utils.instantiate(cfg, grouped_parameters)

    def scheduler(self, cfg: Any, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        return hydra.utils.instantiate(cfg, optimizer=optimizer)
