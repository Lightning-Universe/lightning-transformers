import logging

import pytorch_lightning as pl
import torch
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase, AutoTokenizer


class Instantiator:
    def __getattr__(self, _):
        raise NotImplementedError


class HydraInstantiator(Instantiator):
    def model(self, cfg: DictConfig) -> torch.nn.Module:
        return get_class(cfg.downstream_model_type).from_pretrained(cfg.pretrained_model_name_or_path)

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

    def datamodule(self, cfg: DictConfig, tokenizer: PreTrainedTokenizerBase) -> pl.LightningDataModule:
        return instantiate(cfg, tokenizer=tokenizer)

    def tokenizer(self, cfg: DictConfig) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=cfg.pretrained_model_name_or_path, use_fast=cfg.task.dataset.use_fast
        )

    def logger(self, cfg: DictConfig) -> logging.Logger:
        if cfg.log:
            return instantiate(cfg.logger)

    # def downstream_model(
    #    self,
    #    cfg: DictConfig,
    #    backbone: DictConfig,
    #    optimizer: DictConfig,
    #    scheduler: DictConfig,
    #    config_data_args: DictConfig,
    # ) -> "TODO":
    #    return instantiate(
    #        cfg,
    #        backbone=backbone,
    #        optimizer=optimizer,
    #        scheduler=scheduler,
    #        config_data_args=config_data_args,
    #        # disable hydra instantiation for model to configure optimizer/schedulers
    #        _recursive_=False,
    #    )
