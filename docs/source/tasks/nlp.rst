.. _nlp-tasks:

NLP Tasks
=========

Below are all of the NLP tasks that are supported. We use `HuggingFace Transformers <https://github.com/huggingface/transformers>`_ to create the Transformer model and tokenizers, whilst relying on
`Pytorch Lightning <https://www.pytorchlightning.ai/>`_ for training.

Relying on the Lightning Trainer allows us to use a comprehensive set of tools, such as
training optimizations, multi-node support, powerful override functionalities provided by the LightningModule and various Callbacks such as Stochastic Weight Averaging/Pruning.

Using Lightning also allows us to support custom fine-tuning strategies which can be seen in <TODO>.

.. include:: nlp/language_modeling.rst

.. include:: nlp/multiple_choice.rst

.. include:: nlp/question_answering.rst

.. include:: nlp/summarization.rst

Text Classification
-------------------

Token Classification
--------------------

Translation
-----------
