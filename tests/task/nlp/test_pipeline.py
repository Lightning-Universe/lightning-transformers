import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer

from lightning_transformers.core.nlp import HFBackboneConfig
from lightning_transformers.task.nlp.language_modeling import LanguageModelingDataModule, LanguageModelingTransformer
from lightning_transformers.task.nlp.language_modeling.config import LanguageModelingDataConfig


def test_kwargs_load_from_checkpoint(hf_cache_path, tmpdir):
    """Test to ensure we can pass arguments to hf_pipeline when loading from checkpoint."""

    class TestModel(LanguageModelingTransformer):
        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=1e-5)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="sshleifer/tiny-gpt2")
    model = TestModel(backbone=HFBackboneConfig(pretrained_model_name_or_path="sshleifer/tiny-gpt2"))
    dm = LanguageModelingDataModule(
        cfg=LanguageModelingDataConfig(
            batch_size=1,
            dataset_name="wikitext",
            dataset_config_name="wikitext-2-raw-v1",
            cache_dir=hf_cache_path,
        ),
        tokenizer=tokenizer,
    )
    trainer = pl.Trainer(default_root_dir=tmpdir, fast_dev_run=True)

    trainer.fit(model, dm)

    trainer.save_checkpoint("test.pt")
    kwargs = {"device": 0}
    model = TestModel.load_from_checkpoint("test.pt", hf_pipeline_kwargs=kwargs)
    # todo: refactor this to actually mock the hf_pipeline and assert the input
    assert model._hf_pipeline_kwargs == kwargs
