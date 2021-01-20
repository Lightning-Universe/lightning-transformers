import logging
from typing import Optional, Union

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

from lightning_transformers.core import TransformerDataModule
from lightning_transformers.core.data import TransformerTokenizerDataModule
from lightning_transformers.core.hydra_model import HydraTaskTransformer


class Instantiator:
    def __getattr__(self, _):
        raise NotImplementedError


class HydraInstantiator(Instantiator):
    def model(self, cfg: DictConfig, model_data_args) -> HydraTaskTransformer:
        return instantiate(cfg, **model_data_args)

    def data_module(
        self, cfg: DictConfig, tokenizer: Optional[DictConfig]
    ) -> Union[TransformerDataModule, TransformerTokenizerDataModule]:
        if tokenizer:
            return instantiate(cfg, tokenizer=instantiate(tokenizer))
        return instantiate(cfg)

    def logger(self, cfg: DictConfig) -> logging.Logger:
        if cfg.log:
            return instantiate(cfg.logger)

    def trainer(self, cfg: DictConfig, **kwargs) -> pl.Trainer:
        return instantiate(cfg, **kwargs)
