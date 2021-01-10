import os

import hydra
from hydra.utils import get_class
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.utilities.distributed import rank_zero_info

from lightning_transformers.core.utils import set_ignore_warnings
from lightning_transformers.huggingface import HFTransformerDataModule
from lightning_transformers.huggingface.instantiator import HydraInstantiator


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

    model = get_class(cfg.task._target_)(instantiator, cfg)
    trainer = instantiator.trainer(cfg.trainer, logger=instantiator.logger(cfg))

    if cfg.training.do_train:
        trainer.fit(model, datamodule=data_module)

    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
