Customizing Datasets
====================

Below we describe how to extend the base data module classes to implement your own data processing logic.

This is useful when you have specific data processing you'd like to apply when training/validating or testing using a task, or would like to modify how data is loaded/transformed for the model.

Currently we have examples for two tasks (one encoder, one encoder/decoder), however we will continue to grow the examples over time.

.. include:: nlp/language_modeling_data.rst

.. include:: nlp/translation_data.rst
