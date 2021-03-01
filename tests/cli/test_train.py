import shutil
import subprocess
from unittest import mock
from unittest.mock import MagicMock

from omegaconf import DictConfig

import lightning_transformers.cli.train as cli


def test_train_run():
    trainer_mock = MagicMock()
    instantiator = MagicMock()
    instantiator.trainer.return_value = trainer_mock

    cli.run(instantiator)

    assert trainer_mock.fit.called
    assert trainer_mock.test.called


@mock.patch("lightning_transformers.cli.train.run")
def test_train_main(run_mock):
    cfg = DictConfig({"training": {"do_train": True}})
    cli.main(cfg)
    assert run_mock.called_with(do_train=False, tokenizer=None)


def test_train_entry_point():
    proc = subprocess.run(
        [shutil.which("pl-transformers-train"), "--help"],
        stdout=subprocess.PIPE,
    )
    help = proc.stdout.decode().strip()
    assert help.startswith("train is powered by Hydra")
