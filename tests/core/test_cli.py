from importlib.util import find_spec
from unittest import mock

import pytest
from pytorch_lightning.cli import LightningCLI

from lightning_transformers.task.vision.image_classification import (
    ImageClassificationDataModule,
    ImageClassificationTransformer,
)


@pytest.mark.skipif(find_spec("jsonargparse") is None, reason="jsonargparse is required")
def test_lightning_cli_image_classification():
    config = {
        "data": {
            "dataset_name": "beans",  # Resolve from TransformerDataModule.__init__
        },
        "model": {
            "pretrained_model_name_or_path": "nateraw/tiny-vit-random",  # Resolve from TaskTransformer.__init__
        },
    }
    with mock.patch("sys.argv", ["any.py", f"--config={config}"]):
        cli = LightningCLI(
            ImageClassificationTransformer,
            ImageClassificationDataModule,
            run=False,
        )
    assert cli.config.data.dataset_name == "beans"
    assert cli.config.model.pretrained_model_name_or_path == "nateraw/tiny-vit-random"
    assert isinstance(cli.config_init.data, ImageClassificationDataModule)
    assert isinstance(cli.config_init.model, ImageClassificationTransformer)
