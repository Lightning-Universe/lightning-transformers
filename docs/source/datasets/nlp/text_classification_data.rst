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

    python train.py task=nlp/text_classification dataset.cfg.train_file=train.json dataset.cfg.validation_file=valid.json
