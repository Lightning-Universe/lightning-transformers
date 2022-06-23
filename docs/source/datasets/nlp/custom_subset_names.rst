Custom Subset Names (Edge Cases such as MNLI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some datasets, such as MNLI when loaded from the Huggingface `datasets` library, have special subset names that don't match the standard train/validation/test convention.
Specifically, MNLI has two validation and two test sets, with flavors 'matched' and 'mismatched'.
When using such datasets, you must manually indicate which subset names you want to use for each of train/validation/text.

An example for how to train and validate on MNLI would the the following:

.. code-block:: python

    from lightning_transformers.task.nlp.text_classification import (
        TextClassificationDataModule,
        TextClassificationTransformer,
    )

    dm = TextClassificationDataModule(
        batch_size=1,
        dataset_name="glue",
        dataset_config_name="mnli",
        max_length=512,
        validation_subset_name="validation_matched"
        tokenizer=tokenizer,
    )
