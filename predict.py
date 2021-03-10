"""The shell entry point `$ pl-transformers-predict` is also available"""
import hydra
from omegaconf import DictConfig

from lightning_transformers.cli.predict import main


@hydra.main(config_path="./conf", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
