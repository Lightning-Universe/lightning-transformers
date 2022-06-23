Question Answering Using Your Own Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use custom text files, the files should contain new line delimited json objects within the text files.

The format varies from dataset to dataset as input columns may differ, as well as pre-processing. To make our life easier, we use the SWAG dataset format and override the files that are loaded.

.. code-block:: json

    {
        "answers": {
            "answer_start": [1],
            "text": ["This is a test text"]
        },
        "context": "This is a test context.",
        "id": "1",
        "question": "Is this a test?",
        "title": "train test"
    }

We override the dataset files, allowing us to still use the data transforms defined with this dataset.

.. code-block:: python


    from lightning_transformers.task.nlp.question_answering import (
        SquadDataModule,
    )

    dm = SquadDataModule(
        batch_size=1,
        dataset_config_name="plain_text",
        max_length=384,
        version_2_with_negative=False,
        null_score_diff_threshold=0.0,
        doc_stride=128,
        n_best_size=20,
        max_answer_length=30,
        train_file="path/train.csv",
        validation_file="/path/valid.csv"
        tokenizer=tokenizer,
    )
