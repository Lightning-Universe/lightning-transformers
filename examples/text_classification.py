import pytorch_lightning as pl
from transformers import AutoTokenizer

from lightning_transformers.core.nlp import HFTransformerDataConfig
from lightning_transformers.task.nlp.text_classification import (
    TextClassificationDataModule,
    TextClassificationTransformer,
)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-uncased")
    dm = TextClassificationDataModule(
        cfg=HFTransformerDataConfig(
            batch_size=1,
            dataset_name="glue",
            dataset_config_name="sst2",
            max_length=512,
        ),
        tokenizer=tokenizer,
    )
    dm.setup("fit")
    model = TextClassificationTransformer(pretrained_model_name_or_path="bert-base-uncased", num_labels=dm.num_classes)
    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=1)

    trainer.fit(model, dm)
