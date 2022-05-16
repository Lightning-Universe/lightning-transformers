import pytorch_lightning as pl
from transformers import AutoTokenizer

from lightning_transformers.task.nlp.multiple_choice import (
    MultipleChoiceDataConfig,
    MultipleChoiceTransformer,
    SwagMultipleChoiceDataModule,
)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-uncased")
    model = MultipleChoiceTransformer(pretrained_model_name_or_path="bert-base-uncased")
    dm = SwagMultipleChoiceDataModule(
        cfg=MultipleChoiceDataConfig(
            batch_size=1,
            dataset_name="swag",
            dataset_config_name="regular",
            padding=False,
        ),
        tokenizer=tokenizer,
    )
    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=1)

    trainer.fit(model, dm)
