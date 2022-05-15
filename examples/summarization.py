import pytorch_lightning as pl
from transformers import AutoTokenizer

from lightning_transformers import (
    Seq2SeqDataConfig,
    SummarizationConfig,
    SummarizationTransformer,
    XsumSummarizationDataModule,
)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="t5-base")
    model = SummarizationTransformer(
        pretrained_model_name_or_path="t5-base",
        cfg=SummarizationConfig(
            use_stemmer=True,
            val_target_max_length=142,
            num_beams=None,
            compute_generate_metrics=True,
        ),
    )
    dm = XsumSummarizationDataModule(
        cfg=Seq2SeqDataConfig(
            batch_size=1,
            dataset_name="xsum",
            max_source_length=128,
            max_target_length=128,
        ),
        tokenizer=tokenizer,
    )
    trainer = pl.Trainer(accelerator="auto", devices=1, max_epochs=1)

    trainer.fit(model, dm)
