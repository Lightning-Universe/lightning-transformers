import os

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.distributed import rank_zero_info

from lightning_transformers.core import TaskTransformer
from lightning_transformers.core.config import TrainerConfig
from lightning_transformers.core.huggingface.config import HFTaskConfig, HFTokenizerConfig, HFTransformerDataConfig
from lightning_transformers.core.instantiator import HydraInstantiator, Instantiator
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

    instantiator = HydraInstantiator()

    data_module: TransformerDataModule = instantiator.data_module(cfg=cfg.dataset, tokenizer=cfg.tokenizer)
    data_module.setup("fit")

    model: TaskTransformer = instantiator.model(cfg=cfg.task, model_data_args=data_module.model_data_args)
    trainer = instantiator.trainer(cfg.trainer, logger=instantiator.logger(cfg))

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
