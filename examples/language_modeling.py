import pytorch_lightning as pl
from transformers import AutoTokenizer

from lightning_transformers.task.nlp.language_modeling import (
    LanguageModelingDataConfig,
    LanguageModelingDataModule,
    LanguageModelingTransformer,
)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="gpt2")
    model = LanguageModelingTransformer(pretrained_model_name_or_path="gpt2")
    dm = LanguageModelingDataModule(
        cfg=LanguageModelingDataConfig(batch_size=1, dataset_name="wikitext", dataset_config_name="wikitext-2-raw-v1"),
        tokenizer=tokenizer,
    )
    trainer = pl.Trainer(accelerator="auto", devices="auto")

    trainer.fit(model, dm)
