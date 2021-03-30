.. PyTorchLightning-Sandbox documentation master file, created by
   sphinx-quickstart on Wed Mar 25 21:34:07 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Lightning Transformers
======================

.. include:: introduction.rst

.. toctree::
   :maxdepth: 1
   :name: start
   :caption: Get started

   introduction
   quickstart
   structure/conf

.. toctree::
   :maxdepth: 2
   :name: tasks
   :caption: Tasks

   tasks/nlp/language_modeling
   tasks/nlp/multiple_choice
   tasks/nlp/question_answering
   tasks/nlp/summarization
   tasks/nlp/text_classification
   tasks/nlp/token_classification
   tasks/nlp/translation

.. toctree::
   :maxdepth: 2
   :name: optimization
   :caption: Training Optimizations

   optimizations/deepspeed
   optimizations/sharded

.. toctree::
   :maxdepth: 2
   :name: advanced
   :caption: Advanced

   datasets/custom_data
   advanced/custom_datasets
   advanced/new_task

.. toctree::
   :maxdepth: 1
   :name: api
   :caption: API Reference


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
