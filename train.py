import os

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.distributed import rank_zero_info

from lightning_transformers.core import TaskTransformer
from lightning_transformers.core.huggingface import HFTransformerDataModule
from lightning_transformers.core.huggingface.instantiator import HydraInstantiator
from lightning_transformers.core.utils import set_ignore_warnings


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.ignore_warnings:
        set_ignore_warnings()

    rank_zero_info(OmegaConf.to_yaml(cfg))

    os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"

    instantiator = HydraInstantiator()

    data_module: HFTransformerDataModule = instantiator.data_module(
        cfg.dataset, tokenizer=instantiator.tokenizer(cfg.tokenizer)
    )
    data_module.setup('fit')

    # save some model arguments which are only known dynamically.
    # the instantiator will use them to instantiate the backbone
    instantiator.state["backbone"] = data_module.config_data_args

    model: TaskTransformer = instantiator.model(cfg.task)
    trainer = instantiator.trainer(cfg.trainer, logger=instantiator.logger(cfg))

    if cfg.training.do_train:
        trainer.fit(model, datamodule=data_module)

    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
