import os

import hydra
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.utilities.distributed import rank_zero_info

from lightning_transformers.core.utils import set_ignore_warnings
from lightning_transformers.core.huggingface import HFTransformerDataModule, HFTransformer
from lightning_transformers.core.huggingface.instantiator import HydraInstantiator


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
    data_module.setup()

    # save some model arguments which are only known dynamically.
    # the instantiator will use them to instantiate the backbone
    instantiator.state["backbone"] = data_module.config_data_args

    model: HFTransformer = instantiator.model(cfg.task)
    trainer = instantiator.trainer(cfg.trainer, logger=instantiator.logger(cfg))

    if cfg.training.do_train:
        trainer.fit(model, datamodule=data_module)

    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
