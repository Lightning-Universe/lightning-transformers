import logging
from typing import Optional

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from lightning_transformers.core import TransformerDataModule
from lightning_transformers.core.model import TaskTransformer


class Instantiator:
    def __getattr__(self, _):
        raise NotImplementedError


class HydraInstantiator(Instantiator):

    def model(self,
              task_cfg: DictConfig, model_data_args) -> TaskTransformer:
        return instantiate(task_cfg, self, **model_data_args)

    def optimizer(self, model: torch.nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
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
        return instantiate(cfg, grouped_parameters)

    def scheduler(self, cfg: DictConfig, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        return instantiate(cfg, optimizer=optimizer)

    # todo: signature is wrong here, we might want to create a TransformerTextDataModule which takes a tokenizer.
    def data_module(self, cfg: DictConfig, tokenizer: Optional[DictConfig]) -> TransformerDataModule:
        if tokenizer:
            return instantiate(cfg, tokenizer=instantiate(tokenizer))
        return instantiate(cfg)

    def logger(self, cfg: DictConfig) -> logging.Logger:
        if cfg.log:
            return instantiate(cfg.logger)

    def trainer(self, cfg: DictConfig, **kwargs) -> pl.Trainer:
        return instantiate(cfg, **kwargs)
