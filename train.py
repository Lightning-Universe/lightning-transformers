import os
from dataclasses import dataclass

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.distributed import rank_zero_info

from lightning_transformers.core import TaskTransformer
from lightning_transformers.core.config import TrainerConfig
from lightning_transformers.core.huggingface import HFTransformerDataModule
from lightning_transformers.core.huggingface.config import HFTaskConfig, HFTokenizerConfig, HFTransformerDataConfig
from lightning_transformers.core.huggingface.instantiator import HydraInstantiator, Instantiator
from lightning_transformers.core.utils import set_ignore_warnings


def run(
    instantiator: Instantiator,
    ignore_warnings: bool = True,
    do_train: bool = True,
    dataset: HFTransformerDataConfig = HFTransformerDataConfig(),
    tokenizer: HFTokenizerConfig = HFTokenizerConfig(),
    task: HFTaskConfig = HFTaskConfig(),
    trainer: TrainerConfig = TrainerConfig(),
):
    if ignore_warnings:
        set_ignore_warnings()

    os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"

    data_module: HFTransformerDataModule = instantiator.data_module(
        dataset, tokenizer=instantiator.tokenizer(tokenizer)
    )
    data_module.setup("fit")

    # save some model arguments which are only known dynamically.
    # the instantiator will use them to instantiate the backbone
    instantiator.state["backbone"] = data_module.config_data_args

    model: TaskTransformer = instantiator.model(task)
    trainer = instantiator.trainer(trainer)

    if do_train:
        trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    rank_zero_info(OmegaConf.to_yaml(cfg))
    instantiator = HydraInstantiator()
    cfg = instantiator.dictconfig_to_dataclass(cfg)
    print(cfg)
    run(
        instantiator,
        ignore_warnings=cfg.ignore_warnings,
        do_train=cfg.do_train,
        dataset=cfg.dataset,
        tokenizer=cfg.tokenizer,
        task=cfg.task,
        trainer=cfg.trainer,
    )


if __name__ == "__main__":
    main()
