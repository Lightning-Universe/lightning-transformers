.. _token_classification:

Token Classification
--------------------

The Task
^^^^^^^^
The Token classification Task is similar to text classification, except each token within the text receives a prediction.
A common use of this task is Named Entity Recognition (NER). Use this task if you require your data to be classified at the token level.

Datasets
^^^^^^^^
Currently supports the `conll <https://huggingface.co/datasets/conll2003>`_ dataset, or custom input files.

Training
^^^^^^^^

.. code-block:: python

    import pytorch_lightning as pl
    from transformers import AutoTokenizer

    from lightning_transformers.task.nlp.token_classification import (
        TokenClassificationDataModule,
        TokenClassificationTransformer,
    )

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
    model = TokenClassificationTransformer(pretrained_model_name_or_path="bert-base-uncased", labels=dm.num_classes)
    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=1)

    trainer.fit(model, dm)

We report the Precision, Recall, Accuracy and Cross Entropy Loss for validation.

.. include:: /datasets/nlp/token_classification_data.rst

Token Classification Inference Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default we use the NER pipeline, which requires a an input sequence string and the number of labels.

.. code-block:: python

    from transformers import AutoTokenizer
    from lightning_transformers.task.nlp.token_classification import TokenClassificationTransformer

    model = TokenClassificationTransformer(
        pretrained_model_name_or_path="prajjwal1/bert-tiny",
        tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path="prajjwal1/bert-tiny"),
        labels=2,
    )
    model.hf_predict("Have a good day!")
