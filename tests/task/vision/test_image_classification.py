import sys
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
import transformers
from transformers import AutoFeatureExtractor

from lightning_transformers.task.vision.image_classification import (
    ImageClassificationDataConfig,
    ImageClassificationDataModule,
    ImageClassificationTransformer,
)


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
@pytest.mark.skipif(sys.platform == "darwin", reason="Currently darwin is not working")
def test_smoke_train(hf_cache_path):
    feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path="nateraw/tiny-vit-random")
    dm = ImageClassificationDataModule(
        cfg=ImageClassificationDataConfig(batch_size=1, dataset_name="beans"),
        feature_extractor=feature_extractor,
    )
    model = ImageClassificationTransformer(
        pretrained_model_name_or_path="nateraw/tiny-vit-random", num_labels=dm.num_classes
    )

    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model, dm)


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_predict():
    model = ImageClassificationTransformer(
        pretrained_model_name_or_path="nateraw/tiny-vit-random",
        tokenizer=AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path="nateraw/tiny-vit-random"),
        num_labels=3,
    )
    # predict on the logo
    y = model.hf_predict(
        "https://github.com/PyTorchLightning/lightning-transformers/blob/master/"
        "docs/source/_static/images/logo.png?raw=true"
    )
    assert len(y) == 3


def test_model_has_correct_cfg():
    model = ImageClassificationTransformer(
        pretrained_model_name_or_path="nateraw/tiny-vit-random",
    )
    assert isinstance(model.model, transformers.ViTForImageClassification)


def test_datamodule_has_correct_cfg():
    feature_extractor = MagicMock()
    dm = ImageClassificationDataModule(feature_extractor)
    assert isinstance(dm.cfg, ImageClassificationDataConfig)
    assert dm.tokenizer is feature_extractor
