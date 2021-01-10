from dataclasses import dataclass

from omegaconf import DictConfig

from lightning_transformers.core.model import LitTransformer
from lightning_transformers.huggingface.instantiator import Instantiator


@dataclass
class TransformerConfig:
    model: DictConfig
    optimizer: DictConfig
    scheduler: DictConfig


class HFLitTransformer(LitTransformer):
    def __init__(self, instantiator: Instantiator, model: DictConfig, optimizer: DictConfig, scheduler: DictConfig):
        model = instantiator.model(model)
        optimizer = instantiator.optimizer(model, optimizer)
        scheduler = instantiator.scheduler(scheduler, optimizer)
        super().__init__(model, optimizer, scheduler)
