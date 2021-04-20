Translation Using Your Own Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use custom text files, the files should contain new line delimited json objects within the text files.

.. code-block:: json

    {
        "source": "example source text",
        "target": "example target text"
    }

We override the dataset files, allowing us to still use the data transforms defined with this dataset.

.. code-block:: python

    python train.py task=nlp/translation dataset.cfg.train_file=train.json dataset.cfg.validation_file=valid.json
