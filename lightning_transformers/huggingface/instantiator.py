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
    def model(self, config: DictConfig) -> torch.nn.Module:
        return get_class(config.downstream_model_type).from_pretrained(config.pretrained_model_name_or_path)

    def optimizer(self, model: torch.nn.Module, config: DictConfig) -> torch.optim.Optimizer:
        no_decay = ["bias", "LayerNorm.weight"]
        grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return instantiate(config, grouped_parameters)

    def scheduler(self, config: DictConfig, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        return instantiate(config, optimizer=optimizer)

    def datamodule(self, config: DictConfig, tokenizer: PreTrainedTokenizerBase) -> pl.LightningDataModule:
        return instantiate(config, tokenizer=tokenizer)

    def instantiate_tokenizer(self, config: DictConfig) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=config.pretrained_model_name_or_path, use_fast=config.task.dataset.use_fast
        )

    def logger(self, config: DictConfig) -> logging.Logger:
        if config.log:
            return instantiate(config.logger)

    # def downstream_model(
    #    self,
    #    config: DictConfig,
    #    backbone: DictConfig,
    #    optimizer: DictConfig,
    #    scheduler: DictConfig,
    #    config_data_args: DictConfig,
    # ) -> "TODO":
    #    return instantiate(
    #        config,
    #        backbone=backbone,
    #        optimizer=optimizer,
    #        scheduler=scheduler,
    #        config_data_args=config_data_args,
    #        # disable hydra instantiation for model to configure optimizer/schedulers
    #        _recursive_=False,
    #    )
