Text Classification Using Your Own Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use custom text files, the files should contain new line delimited json objects within the text files.

The label mapping is automatically generated from the training dataset labels if no mapping is given.

.. code-block:: json

    {
        "label": "sad",
        "text": "I'm feeling quite sad and sorry for myself but I'll snap out of it soon."
    }

.. code-block:: python

    from lightning_transformers.task.nlp.text_classification import (
        TextClassificationDataModule,
        TextClassificationTransformer,
    )

    dm = TextClassificationDataModule(
        batch_size=1,
        max_length=512,
        train_file="path/train.json",
        validation_file="/path/valid.json"
        tokenizer=tokenizer,
    )
