import os
import argparse
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

os.environ["HYDRA_FULL_ERROR"] = "1"
from omegaconf import DictConfig

import torch
import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from lightning_transformers.core.utils import (
    instantiate_model,
    instantiate_data_module,
    initialize_loggers,
    load_configuration
)

def train(cfg):
    log.info(OmegaConf.to_yaml(cfg))

    data_module: pl.LightningDataModule = instantiate_data_module(cfg)
    data_module.setup()
    model: pl.LightningModule = instantiate_model(cfg, data_module)

    loggers: List[pl.callbacks.Callback] = initialize_loggers(cfg)

    trainer: pl.Trainer = instantiate(cfg.trainer, gpus=cfg.gpus, logger=loggers)

    if cfg.training.do_train:
        trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


def main(cfg: DictConfig) -> None:
    train(cfg)

if __name__ == "__main__":
    cfg = load_configuration()
    main(cfg)