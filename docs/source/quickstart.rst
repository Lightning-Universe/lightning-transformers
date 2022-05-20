.. _quickstart:

Quick Start
===========

.. include:: installation.rst


Using Lightning-Transformers
****************************

Lightning Transformers provides LightningModules, LightningDataModules and Strategies to use 🤗 Transformers with the PyTorch Lightning Trainer, supporting tasks such as:ref:`language_modeling`, :ref:`translation` and more. To use, simply:

1. Pick a task to train (`LightningModule`)

2. Pick a dataset (`LightningDataModule`)

3. Use any `PyTorch Lightning parameters and optimizations <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_

Here is an example of training `bert-base-cased <https://huggingface.co/bert-base-cased>`__ on the `CARER <https://huggingface.co/datasets/emotion>`__ emotion dataset using the Text Classification task.

.. code-block:: python

    import pytorch_lightning as pl
    from transformers import AutoTokenizer

    from lightning_transformers.task.nlp.text_classification import (
        TextClassificationDataModule,
        TextClassificationTransformer,
        TextClassificationDataConfig,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="bert-base-cased"
    )
    dm = TextClassificationDataModule(
        cfg=TextClassificationDataConfig(
            batch_size=1,
            dataset_name="emotion",
            max_length=512,
        ),
        tokenizer=tokenizer,
    )
    model = TextClassificationTransformer(pretrained_model_name_or_path="bert-base-cased")

    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=1)

    trainer.fit(model, dm)


Changing the Optimizer
----------------------

Swapping to the RMSProp optimizer:

.. code-block:: python

    import pytorch_lightning as pl
    import torch
    import transformers
    from transformers import AutoTokenizer

    from lightning_transformers.task.nlp.text_classification import (
        TextClassificationDataModule,
        TextClassificationTransformer,
        TextClassificationDataConfig,
    )


    class RMSPropTransformer(TextClassificationTransformer):
        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
            # automatically find the total number of steps we need!
            num_training_steps, num_warmup_steps = self.compute_warmup(self.num_training_steps, num_warmup_steps=0.1)
            scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            }


    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="bert-base-cased"
    )
    dm = TextClassificationDataModule(
        cfg=TextClassificationDataConfig(
            batch_size=1,
            dataset_name="emotion",
            max_length=512,
        ),
        tokenizer=tokenizer,
    )
    model = RMSPropTransformer(pretrained_model_name_or_path="bert-base-cased")

    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=1)

    trainer.fit(model, dm)


Enabling DeepSpeed/Sharded/Mixed Precision and more is super simple through the Lightning Trainer.

.. code-block:: python

    # enable DeepSpeed with 16bit precision
    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=1, strategy='deepspeed', precision=16)

    # enable DeepSpeed ZeRO Stage 3 with BFLOAT16 precision
    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=1, strategy='deepspeed_stage_3_offload', precision="bf16")

    # enable Sharded Training with 16bit precision
    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=1, strategy='ddp_sharded', precision=16)

Inference
---------

.. code-block:: python

    from transformers import AutoTokenizer
    from lightning_transformers.task.nlp.text_classification import TextClassificationTransformer

    model = TextClassificationTransformer(
        pretrained_model_name_or_path="bert-base-uncased",
        tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-uncased"),
    )
    model.hf_predict("Lightning rocks!")
   # Returns [{'label': 'LABEL_0', 'score': 0.545...}]
