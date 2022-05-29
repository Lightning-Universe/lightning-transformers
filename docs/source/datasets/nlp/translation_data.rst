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

    from lightning_transformers.task.nlp.translation import (
        TranslationDataConfig,
        WMT16TranslationDataModule,
    )

    dm = WMT16TranslationDataModule(
        cfg=TranslationDataConfig(
            dataset_name="wmt16",
            # WMT translation datasets: ['cs-en', 'de-en', 'fi-en', 'ro-en', 'ru-en', 'tr-en']
            dataset_config_name="ro-en",
            source_language="en",
            target_language="ro",
            max_source_length=128,
            max_target_length=128,
            train_file="path/train.json",
            validation_file="/path/valid.json"
        ),
        tokenizer=tokenizer,
    )
