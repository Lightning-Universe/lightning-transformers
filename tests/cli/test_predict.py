import shutil
import subprocess
from unittest import mock
from unittest.mock import ANY, MagicMock

from omegaconf import OmegaConf

import lightning_transformers.cli.predict as cli


def test_predict_run():
    model = MagicMock()
    instantiator = MagicMock()
    instantiator.model.return_value = model

    cli.run("test", instantiator)

    model.hf_predict.assert_called_with("test")


@mock.patch("lightning_transformers.cli.predict.run")
def test_predict_main(run_mock):
    cfg = OmegaConf.create({"x": "foo", "task": "bar"})
    cli.main(cfg)
    run_mock.assert_called_with(
        "foo",
        ANY,
        checkpoint_path=None,
        task="bar",
        model_data_kwargs=None,
        tokenizer=None,
        pipeline_kwargs={},
        predict_kwargs={},
    )


def test_predict_entry_point():
    proc = subprocess.run([shutil.which("pl-transformers-predict"), "--help"], stdout=subprocess.PIPE)
    help = proc.stdout.decode().strip()
    assert help.startswith("predict is powered by Hydra")
