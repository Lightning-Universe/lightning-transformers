Custom Subset Names (Edge Cases such as MNLI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some datasets, such as MNLI when loaded from the Huggingface `datasets` library, have special subset names that don't match the standard train/validation/test convention.
Specifically, MNLI has two validation and two test sets, with flavors 'matched' and 'mismatched'.
When using such datasets, you must manually indicate which subset names you want to use for each of train/validation/text.
For this, you can set the config variables `dataset.cfg.train_subset_name`, `dataset.cfg.validation_subset_name` and `dataset.cfg.test_subset_name`.

An example for how to train and validate on MNLI would the the following:

.. code-block:: python

    python train.py task=nlp/text_classification dataset=nlp/text_classification/glue dataset.cfg.dataset_config_name=mnli ++dataset.cfg.validation_subset_name=validation_matched

It also works for train and test subsets, like so:

++dataset.cfg.train_subset_name=name_of_subset
++dataset.cfg.test_subset_name=name_of_subset
