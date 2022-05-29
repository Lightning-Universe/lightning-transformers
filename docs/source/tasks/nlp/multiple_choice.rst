.. _multiple_choice:

Multiple Choice
---------------

The Task
^^^^^^^^
The Multiple Choice task requires the model to decide on a set of options, given a question with optional context.

Similar to the text classification task, the model is fine-tuned on multi-class classification to provide probabilities across all possible answers.
This is useful if the data you'd like the model to predict on requires selecting from a set of answers based on context or questions, where the answers can be variable.
In contrast, use the text classification task if the answers remain static and are not needed to be included during training.

Datasets
^^^^^^^^
Currently supports the `RACE <https://huggingface.co/datasets/race>`_ and `SWAG <https://huggingface.co/datasets/swag>`_ datasets, or custom input files.

.. code-block:: none

    Question: What color is the sky?
    Answers:
        A: Blue
        B: Green
        C: Red

    Model answer: A

Training
^^^^^^^^

.. code-block:: python

    import pytorch_lightning as pl
    from transformers import AutoTokenizer

    from lightning_transformers.task.nlp.multiple_choice import (
        MultipleChoiceDataConfig,
        MultipleChoiceTransformer,
        SwagMultipleChoiceDataModule,
    )

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

We report Cross Entropy Loss, Precision, Recall and Accuracy for validation.

.. include:: /datasets/nlp/multiple_choice_data.rst

Multiple Choice Inference
^^^^^^^^^^^^^^^^^^^^^^^^^

Currently there is no HF pipeline available for this model. Feel free to make an issue or PR if you require this functionality.
