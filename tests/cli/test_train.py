import shutil
import subprocess
from unittest import mock
from unittest.mock import ANY, MagicMock

import pytest
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule

import lightning_transformers.cli.train as cli


def test_train_run_raises():
    instantiator = MagicMock()
    instantiator.data_module.return_value = None
    with pytest.raises(ValueError, match="No dataset found"):
        cli.run(instantiator)

    with pytest.raises(ValueError, match="did not return a DataModule"):
        cli.run(MagicMock())


def test_train_run():
    instantiator = MagicMock()
    instantiator.data_module.return_value = MagicMock(spec=LightningDataModule)

    cli.run(instantiator)

    instantiator.data_module.return_value.setup.assert_called_with("fit")
    trainer_mock = instantiator.trainer.return_value
    trainer_mock.fit.assert_called()
    trainer_mock.test.assert_called()


@mock.patch("lightning_transformers.cli.train.run")
def test_train_main(run_mock):
    cfg = OmegaConf.create({"training": {"run_test_after_fit": True}})
    cli.main(cfg)
    run_mock.assert_called_with(
        ANY,
        ignore_warnings=ANY,
        run_test_after_fit=True,
        dataset=ANY,
        task=ANY,
        trainer=ANY,
        logger=None,
        tokenizer=ANY,
    )


def test_train_entry_point():
    proc = subprocess.run([shutil.which("pl-transformers-train"), "--help"], stdout=subprocess.PIPE)
    help = proc.stdout.decode().strip()
    assert help.startswith("train is powered by Hydra")
