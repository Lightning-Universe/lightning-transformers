import os
from pathlib import Path
from typing import Any, List, Optional

import pytest
from hydra.experimental import compose, initialize
from hydra.test_utils.test_utils import find_parent_dir_containing

from lightning_transformers.cli.predict import main as predict_main
from lightning_transformers.cli.train import main as train_main
from tests import CACHE_PATH  # GitHub Actions use this path to cache datasets.


class ScriptRunner:
    def __init__(self, hf_cache_path: Path) -> None:
        self.hf_cache_path = hf_cache_path

    @staticmethod
    def find_hydra_conf_dir(config_dir: str = "conf") -> str:
        """
        Util function to find the hydra config directory from the main repository for testing.
        Args:
            config_dir: Name of config directory.

        Returns: Relative config path

        """
        parent_dir = find_parent_dir_containing(config_dir, initial_dir=os.path.dirname(__file__))
        relative_conf_dir = os.path.relpath(parent_dir, os.path.dirname(__file__))
        return os.path.join(relative_conf_dir, config_dir)

    def train(self, cmd_args: List[str]) -> None:
        print(f"pl-transformers-train {' '.join(cmd_args)}")
        relative_conf_dir = self.find_hydra_conf_dir()
        with initialize(config_path=relative_conf_dir, job_name="test_app"):
            cfg = compose(config_name="config", overrides=cmd_args)
            train_main(cfg)

    def predict(self, cmd_args: List[str]) -> Any:
        print(f"pl-transformers-predict {' '.join(cmd_args)}")
        relative_conf_dir = self.find_hydra_conf_dir()
        with initialize(config_path=relative_conf_dir, job_name="test_app"):
            cfg = compose(config_name="config", overrides=cmd_args)
            return predict_main(cfg)

    def hf_train(
        self,
        task: str,
        model: str,
        dataset: Optional[str] = None,
        cmd_args: Optional[List[str]] = None,
        max_samples: int = 16,
        num_workers: int = 0,
        fast_dev_run: int = 1,
    ) -> None:
        if cmd_args is None:
            cmd_args = []
        cmd_args.extend(
            [
                f"task=nlp/{task}",
                f"backbone.pretrained_model_name_or_path={model}",
                f"dataset.cfg.limit_train_samples={max_samples}",
                f"dataset.cfg.limit_val_samples={max_samples}",
                f"dataset.cfg.limit_test_samples={max_samples}",
                f"dataset.cfg.cache_dir={self.hf_cache_path}",
                f"training.num_workers={num_workers}",
            ]
        )
        if dataset is not None:
            cmd_args.append(f"dataset=nlp/{task}/{dataset}")
        if fast_dev_run:
            cmd_args.append(f"trainer.fast_dev_run={fast_dev_run}")
        self.train(cmd_args)

    def hf_predict(self, cmd_args: List[str], task: str, model: str) -> Any:
        cmd_args.extend(
            [
                f"task=nlp/{task}",
                f"backbone.pretrained_model_name_or_path={model}",
            ]
        )
        return self.predict(cmd_args)


@pytest.fixture(scope="session")
def hf_cache_path() -> Path:
    datadir = Path(CACHE_PATH)
    return datadir / "huggingface"


@pytest.fixture(scope="session")
def script_runner(hf_cache_path: Path) -> ScriptRunner:
    return ScriptRunner(hf_cache_path)
