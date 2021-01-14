import logging
from typing import Optional

import pytorch_lightning as pl
import torch
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from lightning_transformers.core.huggingface import HFTransformerDataModule
from lightning_transformers.core.huggingface.config import HFBackboneConfig

# FIXME: circular import
# from lightning_transformers.core.huggingface import HFTransformer


class Instantiator:
    def __getattr__(self, _):
        raise NotImplementedError


class HydraInstantiator(Instantiator):
    def __init__(self):
        self._state = {}

    def model(self, cfg: DictConfig):  # -> HFTransformer:
        return instantiate(cfg, self)

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

    def data_module(
        self, cfg: DictConfig, tokenizer: Optional[PreTrainedTokenizerBase] = None
    ) -> HFTransformerDataModule:
        return instantiate(cfg, tokenizer=tokenizer)

    # todo: These are HF specific instantiation not Hydra
    # todo: most of this code should live in a base class outside core/huggingface/
    def backbone(self, downstream_model_type: str, backbone_cfg: HFBackboneConfig) -> torch.nn.Module:
        return get_class(downstream_model_type).from_pretrained(
            backbone_cfg.pretrained_model_name_or_path, **self._state["backbone"]
        )

    def tokenizer(self, cfg: HFBackboneConfig) -> PreTrainedTokenizerBase:
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=cfg.pretrained_model_name_or_path, use_fast=cfg.use_fast
        )

    def logger(self, cfg: DictConfig) -> logging.Logger:
        if cfg.log:
            return instantiate(cfg.logger)

    def trainer(self, cfg: DictConfig, **kwargs) -> pl.Trainer:
        return instantiate(cfg, **kwargs)

    @property
    def state(self):
        return self._state
