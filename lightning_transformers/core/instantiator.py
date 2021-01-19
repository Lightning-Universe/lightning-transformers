from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING, Union

import pytorch_lightning as pl
import torch
from dacite import Config, from_dict
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

from lightning_transformers.core import TransformerDataModule
from lightning_transformers.core.config import OptimizerConfig, SchedulerConfig, TrainerConfig, TransformerDataConfig
from lightning_transformers.core.data import TransformerTokenizerDataModule
from lightning_transformers.core.model import TaskTransformer

if TYPE_CHECKING:
    # avoid circular imports
    from lightning_transformers.core.huggingface.config import HFTaskConfig, HFTokenizerConfig


class Instantiator:
    def __getattr__(self, _):
        raise NotImplementedError


class HydraInstantiator(Instantiator):
    # TODO: TaskConfig instead?
    def model(self, cfg: "HFTaskConfig", model_data_args: Dict[str, Any]) -> TaskTransformer:
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

    def dictconfig_to_dataclass(self, cfg: DictConfig, strict: bool = True) -> Union[DictConfig, dataclass]:
        # resolve interpolations
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        converted = self._recursive_convert(cfg, strict=True)
        return converted

    def _recursive_convert(self, cfg: DictConfig, strict: bool = True) -> Union[DictConfig, dataclass]:
        for k, v in cfg.items():
            if isinstance(v, DictConfig):
                cfg[k] = self._recursive_convert(v)
        target = cfg.get("_target_config_")
        if target:
            return from_dict(
                data_class=get_class(target), data=OmegaConf.to_container(cfg), config=Config(strict=strict)
            )
        return cfg
