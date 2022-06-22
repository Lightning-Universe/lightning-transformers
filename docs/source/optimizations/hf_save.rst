.. _hf_save:

HuggingFace Hub Checkpoints
===========================

Lightning Transformers default behaviour means we save PyTorch based checkpoints.

HuggingFace Transformers provides a separate API for saving checkpoints. Below we describe two ways to save HuggingFace checkpoints manually or during training.

To manually save checkpoints from your model:

.. code-block:: python

   model = TextClassificationTransformer(pretrained_model_name_or_path="prajjwal1/bert-tiny")

   # saves a HF checkpoint to this path.
   model.save_hf_checkpoint("checkpoint")

To save an additional HF Checkpoint everytime the checkpoint callback saves, pass in the ``HFSaveCheckpoint`` plugin:

.. code-block:: python

    import pytorch_lightning as pl
    from transformers import AutoTokenizer

    from lightning_transformers.plugins.checkpoint import HFSaveCheckpoint
    from lightning_transformers.task.nlp.text_classification import (
        TextClassificationDataConfig,
        TextClassificationDataModule,
        TextClassificationTransformer,
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    dm = TextClassificationDataModule(
        cfg=TextClassificationDataConfig(
            batch_size=1,
            dataset_name="glue",
            dataset_config_name="sst2",
            max_length=512,
        ),
        tokenizer=tokenizer,
    )
    model = TextClassificationTransformer(pretrained_model_name_or_path="prajjwal1/bert-tiny")
    trainer = pl.Trainer(plugins=HFSaveCheckpoint(model=model))
    trainer.fit(model, dm)
