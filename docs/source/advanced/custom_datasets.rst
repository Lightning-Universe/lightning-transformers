.. _datasets:

Customizing Datasets
====================

You can use Lightning Transformers task on custom datasets by extending the base `DataModule <https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html>`_ classes to implement your own data processing logic.

This is useful when you have specific data processing you'd like to apply when training/validating or testing using a task, or would like to modify how data is loaded/transformed for the model.

Currently we have examples for two tasks (one encoder, one encoder/decoder), more examples coming soon!

.. include:: nlp/custom_subset_names.rst

.. include:: nlp/language_modeling_data.rst

.. include:: nlp/translation_data.rst
