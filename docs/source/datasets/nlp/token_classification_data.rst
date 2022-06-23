Token Classification Using Your Own Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use custom text files, the files should contain new line delimited json objects within the text files. For each token, there should be an associated label.

.. code-block:: json

    {
        "label_tags": [11, 12, 12, 21, 13, 11, 11, 21, 13, 11, 12, 13, 11, 21, 22, 11, 12, 17, 11, 21, 17, 11, 12, 12, 21, 22, 22, 13, 11, 0],
        "tokens": ["The", "European", "Commission", "said", "on", "Thursday", "it", "disagreed", "with", "German", "advice", "to", "consumers"]
    }

.. code-block:: python

    from lightning_transformers.task.nlp.token_classification import TokenClassificationDataModule

    dm = TokenClassificationDataModule(
        batch_size=1,
        task_name="ner",
        dataset_name="conll2003",
        preprocessing_num_workers=1,
        label_all_tokens=False,
        revision="master",
        train_file="path/train.json",
        validation_file="/path/valid.json"
        tokenizer=tokenizer,
    )
