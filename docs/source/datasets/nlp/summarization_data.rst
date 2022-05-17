Summarization Using Your Own Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use custom text files, the files should contain new line delimited json objects within the text files.

.. code-block:: json

    {
        "source": "some-body",
        "target": "some-sentence"
    }

We override the dataset files, allowing us to still use the data transforms defined with this dataset.

.. code-block:: python

    from lightning_transformers.task.nlp.summarization import (
        SummarizationConfig,
        SummarizationDataConfig,
        XsumSummarizationDataModule,
    )

    dm = XsumSummarizationDataModule(
        cfg=SummarizationDataConfig(
            batch_size=1,
            dataset_name="xsum",
            max_source_length=128,
            max_target_length=128,
            train_file="path/train.csv",
            validation_file="/path/valid.csv"
        ),
        tokenizer=tokenizer,
    )
