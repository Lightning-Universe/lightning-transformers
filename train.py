import os
import warnings
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from lightning_transformers.core import TaskTransformer, TransformerDataModule
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_info
from lightning_transformers.core.utils import (
    instantiate_downstream_model,
    instantiate_data_module,
    initialize_loggers
)


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    if cfg.ignore_warnings:
        warnings.simplefilter("ignore")

    rank_zero_info(OmegaConf.to_yaml(cfg))

    os.environ['TOKENIZERS_PARALLELISM'] = 'TRUE'

    logger = initialize_loggers(cfg)

    data_module: TransformerDataModule = instantiate_data_module(
        dataset_config=cfg.dataset,
        training_config=cfg.training,
        tokenizer=cfg.tokenizer
    )
    data_module.setup()

    model: TaskTransformer = instantiate_downstream_model(
        task_config=cfg.task,
        backbone_model_config=cfg.backbone,
        optimizer_config=cfg.optimizer,
        scheduler_config=cfg.scheduler,
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
