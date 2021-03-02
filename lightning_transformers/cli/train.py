from typing import Any, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.distributed import rank_zero_info

from lightning_transformers.core import TaskTransformer, TransformerDataModule
from lightning_transformers.core.config import TaskConfig, TrainerConfig, TransformerDataConfig
from lightning_transformers.core.instantiator import HydraInstantiator, Instantiator
from lightning_transformers.core.nlp.config import TokenizerConfig
from lightning_transformers.core.utils import set_ignore_warnings


def run(
    instantiator: Instantiator,
    ignore_warnings: bool = True,
    do_train: bool = True,
    dataset: TransformerDataConfig = TransformerDataConfig(),
    task: TaskConfig = TaskConfig(),
    trainer: TrainerConfig = TrainerConfig(),
    tokenizer: Optional[TokenizerConfig] = None,
    logger: Optional[Any] = None,
) -> None:
    if ignore_warnings:
        set_ignore_warnings()

    data_module_kwargs = {}
    if tokenizer is not None:
        data_module_kwargs["tokenizer"] = tokenizer

    data_module: TransformerDataModule = instantiator.data_module(dataset, **data_module_kwargs)
    data_module.setup("fit")

    model: TaskTransformer = instantiator.model(task, model_data_args=data_module.model_data_args)
    trainer = instantiator.trainer(trainer, logger=logger)

    if do_train:
        trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


def main(cfg: DictConfig) -> None:
    rank_zero_info(OmegaConf.to_yaml(cfg))
    instantiator = HydraInstantiator()
    logger = instantiator.logger(cfg)
    run(
        instantiator,
        ignore_warnings=cfg.ignore_warnings,
        do_train=cfg.training.do_train,
        dataset=cfg.dataset,
        tokenizer=cfg.get("tokenizer", None),
        task=cfg.task,
        trainer=cfg.trainer,
        logger=logger,
    )


@hydra.main(config_path="../../conf", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
