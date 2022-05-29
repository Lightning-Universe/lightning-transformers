.. _text_classification:

Text Classification
-------------------

The Task
^^^^^^^^
The Text Classification Task fine-tunes the model to predict probabilities across a set of labels given input text. The task supports both binary and multi-class/multi-label classification.

Datasets
^^^^^^^^
Currently supports the `XLNI <https://huggingface.co/datasets/xlni>`_, `GLUE <https://huggingface.co/datasets/glue>`_ and `emotion <https://huggingface.co/datasets/emotion>`_ datasets, or custom input files.

.. code-block:: none

    Input: I don't like this at all!

    Model answer: {"label": "angry", "score": 0.8}

Training
^^^^^^^^
Use this task when you would like to fine-tune Transformers on a labeled text classification task.
For this task, you can rely on most Transformer models as your backbone.

.. code-block:: python

    import pytorch_lightning as pl
    from transformers import AutoTokenizer

    from lightning_transformers.task.nlp.text_classification import (
        TextClassificationDataConfig,
        TextClassificationDataModule,
        TextClassificationTransformer,
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-uncased")
    dm = TextClassificationDataModule(
        cfg=TextClassificationDataConfig(
            batch_size=1,
            dataset_name="glue",
            dataset_config_name="sst2",
            max_length=512,
        ),
        tokenizer=tokenizer,
    )
    model = TextClassificationTransformer(pretrained_model_name_or_path="bert-base-uncased", num_labels=dm.num_classes)
    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=1)

    trainer.fit(model, dm)

We report the Precision, Recall, Accuracy and Cross Entropy Loss for validation.

.. include:: /datasets/nlp/text_classification_data.rst

Text Classification Inference Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default we use the sentiment-analysis pipeline, which requires an input string.

.. code-block:: python

    from transformers import AutoTokenizer
    from lightning_transformers.task.nlp.text_classification import TextClassificationTransformer

    model = TextClassificationTransformer(
        pretrained_model_name_or_path="prajjwal1/bert-tiny",
        tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path="prajjwal1/bert-tiny"),
    )
    model.hf_predict("Lightning rocks!")
