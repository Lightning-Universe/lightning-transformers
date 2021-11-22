import pytorch_lightning as pl
from transformers import AutoTokenizer

from examples.custom_language_modeling.dataset import MyLanguageModelingDataModule
from examples.custom_language_modeling.model import MyLanguageModelingTransformer
from lightning_transformers.core.nlp import HFBackboneConfig
from lightning_transformers.task.nlp.language_modeling.config import LanguageModelingDataConfig


def test_example(hf_cache_path):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    model = MyLanguageModelingTransformer(
        backbone=HFBackboneConfig(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    )
    dm = MyLanguageModelingDataModule(
        cfg=LanguageModelingDataConfig(
            batch_size=1,
            dataset_name="wikitext",
            dataset_config_name="wikitext-2-raw-v1",
            cache_dir=hf_cache_path,
            preprocessing_num_workers=1,
        ),
        tokenizer=tokenizer,
    )
    trainer = pl.Trainer(fast_dev_run=True)

    trainer.fit(model, dm)
