from dataclasses import dataclass

import torch
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig

from lightning_transformers.core.model import LitTransformer


@dataclass
class TransformerConfig:
    model: DictConfig
    optimizer: DictConfig
    scheduler: DictConfig


class HydraMixin:
    def prepare_model(self, config: DictConfig):
        return get_class(config.downstream_model_type).from_pretrained(config.pretrained_model_name_or_path)

    def prepare_optimizer(self, model: torch.nn.Module, config: DictConfig) -> torch.optim.Optimizer:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return instantiate(config, optimizer_grouped_parameters)

    def prepare_scheduler(
        self, config: DictConfig, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return instantiate(config=config, optimizer=optimizer)


class HFLitTransformer(HydraMixin, LitTransformer):
    def __init__(self, model: DictConfig, optimizer: DictConfig, scheduler: DictConfig):
        model = self.prepare_model(model)
        optimizer = self.prepare_optimzer(model, optimizer)
        scheduler = self.prepare_scheduler(scheduler, optimizer)
        super().__init__(model, optimizer, scheduler)
