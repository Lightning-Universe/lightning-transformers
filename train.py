import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.distributed import rank_zero_info

from lightning_transformers.core.config import OptimizerConfig, TrainerConfig
from lightning_transformers.core.huggingface import HFTransformer, HFTransformerDataModule
from lightning_transformers.core.huggingface.config import (
    HFBackboneConfig,
    HFSchedulerConfig,
    HFTokenizerConfig,
    HFTransformerDataConfig,
)
from lightning_transformers.core.huggingface.instantiator import HydraInstantiator
from lightning_transformers.core.utils import set_ignore_warnings


def run(
    ignore_warnings: bool = True,
    do_train: bool = True,
    dataset_cfg: HFTransformerDataConfig = HFTransformerDataConfig(),
    tokenizer_cfg: HFTokenizerConfig = HFTokenizerConfig(),
    backbone_cfg: HFBackboneConfig = HFBackboneConfig(),
    optimizer_cfg: OptimizerConfig = OptimizerConfig(),
    scheduler_cfg: HFSchedulerConfig = HFSchedulerConfig(),
    trainer_cfg: TrainerConfig = TrainerConfig(),
):
    if ignore_warnings:
        set_ignore_warnings()

    os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"

    instantiator = HydraInstantiator()

    data_module: HFTransformerDataModule = instantiator.data_module(
        dataset_cfg, tokenizer=instantiator.tokenizer(tokenizer_cfg)
    )
    data_module.setup()

    # save some model arguments which are only known dynamically.
    # the instantiator will use them to instantiate the backbone
    instantiator.state["backbone"] = data_module.config_data_args

    model: TaskTransformer = instantiator.model(cfg.task)
    trainer = instantiator.trainer(cfg.trainer)

    if do_train:
        trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    rank_zero_info(OmegaConf.to_yaml(cfg))
    cfg = recursive_convert(cfg)  # convert DictConfigs to dataclasses
    run(**vars(cfg))


if __name__ == "__main__":
    main()
