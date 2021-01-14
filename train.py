import os

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.distributed import rank_zero_info

from lightning_transformers.core import TaskTransformer, TransformerDataModule
from lightning_transformers.core.instantiator import HydraInstantiator
from lightning_transformers.core.utils import set_ignore_warnings


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.ignore_warnings:
        set_ignore_warnings()

    rank_zero_info(OmegaConf.to_yaml(cfg))

    os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"

    instantiator = HydraInstantiator()

    data_module: TransformerDataModule = instantiator.data_module(cfg=cfg.dataset, tokenizer=cfg.tokenizer)
    data_module.setup("fit")

    model: TaskTransformer = instantiator.model(task_cfg=cfg.task, model_data_args=data_module.model_data_args)
    trainer = instantiator.trainer(cfg.trainer, logger=instantiator.logger(cfg))

    if cfg.training.do_train:
        trainer.fit(model, datamodule=data_module)

    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
