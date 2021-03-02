from typing import Any, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.distributed import rank_zero_info

from lightning_transformers.core.config import TaskConfig
from lightning_transformers.core.instantiator import HydraInstantiator, Instantiator
from lightning_transformers.core.nlp.config import TokenizerConfig
from lightning_transformers.core.nlp.huggingface import HFTransformer


def run(
    x: Any,
    instantiator: Instantiator,
    task: TaskConfig = TaskConfig(),
    tokenizer: Optional[TokenizerConfig] = None,
) -> Any:
    # TODO: num classes?
    model: HFTransformer = instantiator.model(task, tokenizer=tokenizer)
    return model.hf_predict(x)


def main(cfg: DictConfig):
    rank_zero_info(OmegaConf.to_yaml(cfg))
    instantiator = HydraInstantiator()
    y = run(cfg.x, instantiator, task=cfg.task, tokenizer=cfg.tokenizer if "tokenizer" in cfg else None)
    rank_zero_info(y)


@hydra.main(config_path="../../conf", config_name="config")
def hydra_entry(cfg: DictConfig):
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
