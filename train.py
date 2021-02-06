import hydra
from omegaconf import DictConfig

from cli import main


@hydra.main(config_path="conf", config_name="config")
def hydra_main(cfg: DictConfig):
    main(cfg)


if __name__ == "__main__":
    hydra_main()
