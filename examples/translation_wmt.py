import pytorch_lightning as pl
from transformers import AutoTokenizer

from lightning_transformers.task.nlp.translation import TranslationTransformer, WMT16TranslationDataModule
from lightning_transformers.task.nlp.translation.config import TranslationConfig, TranslationDataConfig

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="t5-base")
    model = TranslationTransformer(
        pretrained_model_name_or_path="t5-base",
        cfg=TranslationConfig(
            n_gram=4,
            smooth=False,
            val_target_max_length=142,
            num_beams=None,
            compute_generate_metrics=True,
        ),
    )
    dm = WMT16TranslationDataModule(
        cfg=TranslationDataConfig(
            dataset_name="wmt16",
            # WMT translation datasets: ['cs-en', 'de-en', 'fi-en', 'ro-en', 'ru-en', 'tr-en']
            dataset_config_name="ro-en",
            source_language="en",
            target_language="ro",
            max_source_length=128,
            max_target_length=128,
        ),
        tokenizer=tokenizer,
    )
    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=1)

    trainer.fit(model, dm)
