.. _custom-data:

Custom Data Files
=================

In most cases when training/validating/testing on custom files, you’ll be able to do so without modifying any code, using the general data module classes directly.

Below we show per task how to fine-tune/validate/test on your own files per task or modify the logic within the data classes. Some tasks are more involved than others, as they may require more data processing.

.. include:: nlp/custom_subset_names.rst

.. include:: nlp/language_modeling_data.rst

.. include:: nlp/multiple_choice_data.rst

.. include:: nlp/question_answering_data.rst

.. include:: nlp/summarization_data.rst

.. include:: nlp/text_classification_data.rst

.. include:: nlp/token_classification_data.rst

.. include:: nlp/translation_data.rst
