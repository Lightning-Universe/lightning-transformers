from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING, Union

import pytorch_lightning as pl
import torch
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

from lightning_transformers.core import TransformerDataModule
from lightning_transformers.core.config import (
    BaseConfig,
    OptimizerConfig,
    SchedulerConfig,
    TaskConfig,
    TrainerConfig,
    TransformerDataConfig,
)
from lightning_transformers.core.data import TransformerTokenizerDataModule

if TYPE_CHECKING:
    # avoid circular imports
    from lightning_transformers.core import TaskTransformer
    from lightning_transformers.core.nlp.huggingface.config import HFTokenizerConfig


class Instantiator(ABC):
    def model(self, *args, **kwargs):
        raise NotImplementedError

    def optimizer(self, *args, **kwargs):
        raise NotImplementedError

    def scheduler(self, *args, **kwargs):
        raise NotImplementedError

    def data_module(self, *args, **kwargs):
        raise NotImplementedError

    # def logger(self, *args, **kwargs):
    #    raise NotImplementedError

    def trainer(self, *args, **kwargs):
        raise NotImplementedError


class HydraInstantiator(Instantiator):
    def model(self, cfg: TaskConfig, model_data_args: Dict[str, Any]) -> "TaskTransformer":
        return instantiate(cfg, self, **model_data_args)

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
        self, cfg: TransformerDataConfig, tokenizer: Optional["HFTokenizerConfig"]
    ) -> Union[TransformerDataModule, TransformerTokenizerDataModule]:
        if tokenizer:
            return instantiate(cfg, tokenizer=instantiate(tokenizer))
        return instantiate(cfg)

    # def logger(self, cfg: DictConfig) -> logging.Logger:
    #    if cfg.log:
    #        return instantiate(cfg.logger)

    def trainer(self, cfg: TrainerConfig, **kwargs) -> pl.Trainer:
        return instantiate(cfg, **kwargs)

    def dictconfig_to_dataclass(self, cfg: DictConfig) -> dataclass:
        cfg = OmegaConf.to_container(cfg, resolve=True)  # resolve interpolations

        def _recursive_convert(cfg: Dict) -> dataclass:
            for k, v in cfg.items():
                if isinstance(v, dict):
                    cfg[k] = _recursive_convert(v)
            data_class = get_class(cfg["_target_config_"]) if "_target_config_" in cfg else BaseConfig
            try:
                return data_class(**cfg)
            except TypeError as e:
                unexpected_key = str(e).split(" ")[-1]
                raise KeyError(
                    f"Found an unexpected key {unexpected_key} in the config "
                    f"file for target config {data_class.__name__}"
                )

        converted = _recursive_convert(cfg)
        return converted
