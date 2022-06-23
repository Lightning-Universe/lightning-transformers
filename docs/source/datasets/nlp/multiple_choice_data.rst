Multiple Choice Using Your Own Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use custom text files, the files should contain the data you want to train and validate on and be in CSV or JSON format as described below.

The format varies from dataset to dataset as input columns may differ, as well as pre-processing. To make our life easier, we use the RACE dataset format and override the files that are loaded.

Below we have defined a json file to use as our input data.

.. code-block:: json

    {
        "article": "The man walked into the red house but couldn't see where the light was.",
        "question": "What colour is the house?",
        "options": ["White", "Red", "Blue"]
        "answer": "Red"
    }


We override the dataset files, allowing us to still use the data transforms defined with the RACE dataset.

.. code-block:: python

    from lightning_transformers.task.nlp.multiple_choice import (
        RaceMultipleChoiceDataModule,
    )

    dm = RaceMultipleChoiceDataModule(
        batch_size=1,
        dataset_config_name="all",
        padding=False,
        train_file="path/train.json",
        validation_file="/path/valid.json"
        tokenizer=tokenizer,
    )
