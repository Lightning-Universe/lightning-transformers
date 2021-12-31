Language Modeling Using Your Own Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use custom text files, the files should contain the raw data you want to train and validate on.

During data pre-processing the text is flattened, and the model is trained and validated on context windows (block size) made from the input text. We override the dataset files, allowing us to still use the data transforms defined with the base datamodule.

Below we have defined a csv file to use as our input data.

.. code-block::

    text,
    this is the first sentence,
    this is the second sentence,


When specifying the file path with hydra, it is important to use the absolute path to the file.

.. code-block:: python

    python train.py task=nlp/language_modeling dataset.cfg.train_file=abs/path/train.csv dataset.cfg.validation_file=abs/path/valid.csv
