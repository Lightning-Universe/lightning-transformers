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

.. code-block:: bash

    python train.py +task=nlp/question_answering +dataset=nlp/question_answering/squad dataset.train_file=train.json dataset.validation_file=valid.json
