import os

import pytest
from hydra.experimental import compose, initialize
from hydra.test_utils.test_utils import find_parent_dir_containing

from lit_transformers_cli import main


def find_hydra_conf_dir(config_dir="conf"):
    """
    Util function to find the hydra config directory from the main repository for testing.
    Args:
        config_dir: Name of config directory.

    Returns: Relative config path

    """
    parent_dir = find_parent_dir_containing(config_dir)
    relative_conf_dir = os.path.relpath(parent_dir, os.path.dirname(__file__))
    return os.path.join(relative_conf_dir, config_dir)


@pytest.fixture
def hydra_runner():

    def run(task: str, dataset: str, suffix: str = ""):
        cmd_line = f"+task={task} +dataset={dataset} trainer.fast_dev_run=True " + suffix
        relative_conf_dir = find_hydra_conf_dir()
        with initialize(config_path=relative_conf_dir, job_name="test_app"):
            cfg = compose(config_name="config", overrides=cmd_line.split(" "))
            main(cfg)

    return run
