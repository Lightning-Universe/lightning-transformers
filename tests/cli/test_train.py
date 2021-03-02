import shutil
import subprocess
from unittest import mock
from unittest.mock import ANY, MagicMock

from omegaconf import OmegaConf

import lightning_transformers.cli.train as cli


def test_train_run():
    trainer_mock = MagicMock()
    instantiator = MagicMock()
    instantiator.trainer.return_value = trainer_mock

    cli.run(instantiator)

    trainer_mock.fit.assert_called()
    trainer_mock.test.assert_called()


@mock.patch("lightning_transformers.cli.train.run")
def test_train_main(run_mock):
    cfg = OmegaConf.create({"training": {"do_train": True}})
    cli.main(cfg)
    run_mock.assert_called_with(
        ANY, ignore_warnings=ANY, do_train=True, dataset=ANY, task=ANY, trainer=ANY, logger=None, tokenizer=ANY
    )


def test_train_entry_point():
    proc = subprocess.run(
        [shutil.which("pl-transformers-train"), "--help"],
        stdout=subprocess.PIPE,
    )
    help = proc.stdout.decode().strip()
    assert help.startswith("train is powered by Hydra")
