.. _large_model:

DeepSpeed Training with Big Transformer Models
==============================================

Below is an example of how you can run train a 6B parameter transformer model using Lightning Transformers and DeepSpeed.

The below script was tested on an 8 A100 machine.

.. code-block:: python

    import pytorch_lightning as pl
    from transformers import AutoTokenizer

    from lightning_transformers.task.nlp.language_modeling import LanguageModelingDataModule, LanguageModelingTransformer

    model = LanguageModelingTransformer(
        pretrained_model_name_or_path="EleutherAI/gpt-j-6B",
        tokenizer=AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B"),
        deepspeed_sharding=True # defer initialization of the model to shard/load pre-train weights
    )

    dm = LanguageModelingDataModule(
        batch_size=1,
        dataset_name="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        tokenizer=tokenizer,
    )
    trainer = pl.Trainer(accelerator="gpu", devices="auto", strategy="deepspeed_stage_3", precision=16, max_epochs=1)

    trainer.fit(model, dm)

If you have your own `pl.LightningModule` you can use DeepSpeed Stage 3 sharding + Transformers as well, just add this code:

.. code-block:: python

    import pytorch_lightning as pl
    from transformers import T5ForConditionalGeneration
    from lightning_transformers.utilities.deepspeed import enable_transformers_pretrained_deepspeed_sharding


    class MyModel(pl.LightningModule):

        def setup(self, stage: Optional[str] = None) -> None:
            if stage == "fit" and not hasattr(self, "ptlm"):
                enable_transformers_pretrained_deepspeed_sharding(self)
                self.ptlm = T5ForConditionalGeneration.from_pretrained("t5-11b")
