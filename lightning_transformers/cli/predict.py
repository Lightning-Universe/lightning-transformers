from typing import Any, Dict, Optional

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
    checkpoint_path: Optional[str] = None,
    task: TaskConfig = TaskConfig(),
    model_data_args: Optional[Dict[str, Any]] = None,
    tokenizer: Optional[TokenizerConfig] = None,
    **predict_kwargs: Any
) -> Any:
    model: HFTransformer = instantiator.model(task, model_data_args=model_data_args, tokenizer=tokenizer)
    if checkpoint_path:
        model.load_from_checkpoint(checkpoint_path)
    # TODO: fix dict input
    return model.hf_predict(x, **predict_kwargs)


def main(cfg: DictConfig) -> Any:
    rank_zero_info(OmegaConf.to_yaml(cfg))
    instantiator = HydraInstantiator()
    y = run(
        cfg.x,
        instantiator,
        checkpoint_path=cfg.get("checkpoint_path", None),
        task=cfg.task,
        model_data_args=cfg.get("model_data_args", None),
        tokenizer=cfg.get("tokenizer", None),
        **cfg.get("predict_kwargs", {})
    )
    rank_zero_info(y)
    return y


@hydra.main(config_path="../../conf", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
