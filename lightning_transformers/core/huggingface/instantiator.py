from typing import Optional

import torch
from hydra.utils import get_class, instantiate
from pytorch_lightning import Trainer
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from lightning_transformers.core.config import OptimizerConfig, SchedulerConfig, TrainerConfig
from lightning_transformers.core.huggingface import HFTransformer, HFTransformerDataModule
from lightning_transformers.core.huggingface.config import (
    HFBackboneConfig,
    HFTaskConfig,
    HFTokenizerConfig,
    HFTransformerDataConfig,
)


class Instantiator:
    def __getattr__(self, _):
        raise NotImplementedError


class HydraInstantiator(Instantiator):
    def __init__(self):
        self.state = {}

    def model(self, cfg: HFTaskConfig) -> HFTransformer:
        return instantiate(cfg)

    def backbone(self, cfg: HFBackboneConfig) -> torch.nn.Module:
        return get_class(cfg.downstream_model_type).from_pretrained(
            cfg.pretrained_model_name_or_path, **self.state["backbone"]
        )

    def optimizer(self, model: torch.nn.Module, cfg: OptimizerConfig) -> torch.optim.Optimizer:
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

    def scheduler(
        self, cfg: SchedulerConfig, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return instantiate(cfg, optimizer=optimizer)

    def data_module(
        self, cfg: HFTransformerDataConfig, tokenizer: Optional[PreTrainedTokenizerBase] = None
    ) -> HFTransformerDataModule:
        return instantiate(cfg, tokenizer=tokenizer)

    def tokenizer(self, cfg: HFTokenizerConfig) -> PreTrainedTokenizerBase:
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=cfg.pretrained_model_name_or_path, use_fast=cfg.use_fast
        )

    def trainer(self, cfg: TrainerConfig) -> Trainer:
        return instantiate(cfg)
