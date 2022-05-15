import pytorch_lightning as pl
from transformers import AutoTokenizer

from lightning_transformers.task.nlp.masked_language_modeling import (
    MaskedLanguageModelingDataModule,
    MaskedLanguageModelingTransformer,
)
from lightning_transformers.task.nlp.masked_language_modeling.config import MaskedLanguageModelingDataConfig

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-uncased")
    model = MaskedLanguageModelingTransformer(pretrained_model_name_or_path="bert-base-uncased")
    dm = MaskedLanguageModelingDataModule(
        cfg=MaskedLanguageModelingDataConfig(
            batch_size=1,
            dataset_name="wikitext",
            dataset_config_name="wikitext-2-raw-v1",
            max_length=512,
        ),
        tokenizer=tokenizer,
    )
    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=1)

    trainer.fit(model, dm)
