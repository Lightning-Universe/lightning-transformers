import os

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.utilities.distributed import rank_zero_info

from lightning_transformers.core import TaskTransformer, TransformerDataModule
from lightning_transformers.core.utils import (
    instantiate_downstream_model,
    instantiate_data_module,
    initialize_loggers, set_ignore_warnings, instantiate_tokenizer
)


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    if cfg.ignore_warnings:
        set_ignore_warnings()

    rank_zero_info(OmegaConf.to_yaml(cfg))

    os.environ['TOKENIZERS_PARALLELISM'] = 'TRUE'

    logger = initialize_loggers(cfg)

    tokenizer = instantiate_tokenizer(
        cfg=cfg.tokenizer
    )

    data_module: TransformerDataModule = instantiate_data_module(
        dataset_config=cfg.dataset,
        training_config=cfg.training,
        tokenizer=tokenizer
    )
    data_module.setup()

    model: TaskTransformer = instantiate_downstream_model(
        task_config=cfg.task,
        backbone_model_config=cfg.backbone,
        optimizer_config=cfg.optimizer,
        scheduler_config=cfg.scheduler,
        tokenizer=tokenizer,
        config_data_args=data_module.config_data_args
    )

    trainer: pl.Trainer = instantiate(cfg.trainer, logger=logger)

    if cfg.training.do_train:
        trainer.fit(
            model=model,
            datamodule=data_module
        )

    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
