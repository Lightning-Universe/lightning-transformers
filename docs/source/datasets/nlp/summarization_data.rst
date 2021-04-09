Summarization Using Your Own Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use custom text files, the files should contain new line delimited json objects within the text files.

.. code-block:: json

    {
        "source": "some-body",
        "target": "some-sentence"
    }

We override the dataset files, allowing us to still use the data transforms defined with this dataset.

.. code-block:: bash

    python train.py task=nlp/summarization dataset.cfg.train_file=train.json dataset.cfg.validation_file=valid.json
