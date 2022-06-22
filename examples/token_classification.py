import pytorch_lightning as pl
from transformers import AutoTokenizer

from lightning_transformers.task.nlp.token_classification import (
    TokenClassificationDataModule,
    TokenClassificationTransformer,
)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-uncased")
    dm = TokenClassificationDataModule(
        batch_size=1,
        task_name="ner",
        dataset_name="conll2003",
        preprocessing_num_workers=1,
        label_all_tokens=False,
        revision="master",
        tokenizer=tokenizer,
    )
    model = TokenClassificationTransformer(pretrained_model_name_or_path="bert-base-uncased", labels=dm.labels)
    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=1)

    trainer.fit(model, dm)
