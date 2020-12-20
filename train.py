import os

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from lightning_transformers.core import LitTransformer, LitTransformerDataModule

os.environ["HYDRA_FULL_ERROR"] = "1"
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from lightning_transformers.core.utils import (
    instantiate_downstream_model,
    instantiate_data_module
)


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))

    os.environ['TOKENIZERS_PARALLELISM'] = 'TRUE'

    data_module: LitTransformerDataModule = instantiate_data_module(
        dataset_config=cfg.dataset,
        training_config=cfg.training,
        tokenizer=cfg.tokenizer
    )
    data_module.setup()

    model: LitTransformer = instantiate_downstream_model(
        task_config=cfg.task,
        model_config=cfg.model,
        optimizer_config=cfg.optimizer,
        scheduler_config=cfg.scheduler,
        **data_module.data_model_kwargs
    )

    model.tokenizer = data_module.tokenizer

    trainer: pl.Trainer = instantiate(cfg.trainer)

    if cfg.training.do_train:
        trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
