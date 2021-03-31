import pytorch_lightning as pl
from transformers import AutoTokenizer

from examples.custom.dataset.language_modeling.custom_dataset import MyLanguageModelingDataModule
from examples.custom.dataset.language_modeling.custom_model import MyTranslationTransformer
from lightning_transformers.core.nlp.huggingface import HFBackboneConfig
from lightning_transformers.task.nlp.language_modeling.config import LanguageModelingDataConfig


def test_example(hf_cache_path):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='sshleifer/tiny-gpt2')
    model = MyTranslationTransformer(backbone=HFBackboneConfig(pretrained_model_name_or_path='sshleifer/tiny-gpt2'))
    dm = MyLanguageModelingDataModule(
        cfg=LanguageModelingDataConfig(
            batch_size=1,
            dataset_name='wikitext',
            dataset_config_name='wikitext-2-raw-v1',
            cache_dir=hf_cache_path,
        ),
        tokenizer=tokenizer
    )
    trainer = pl.Trainer(fast_dev_run=True)

    trainer.fit(model, dm)
