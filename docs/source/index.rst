.. PyTorchLightning-Sandbox documentation master file, created by
   sphinx-quickstart on Wed Mar 25 21:34:07 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Lightning-Transformers documentation
=======================================

Lightning Transformers provides capabilities for high performance research using SOTA Transformers, backed by Pytorch Lightning.

We provide an easy to use API across multiple modalities such as NLP and Vision whilst making it simple to add your own
`LightningModules <https://pytorch-lightning.readthedocs.io/en/stable/lightning_module.html#lightning-module>`_ and `LightningDataModules <https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html#lightningdatamodule>`_.

We also provide flexible and easy to enable training optimizations, such as `DeepSpeed ZeRO <https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html#deepspeed>`_ or `FairScale Sharded Training <https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html#sharded-training>`_ + integrations with training optimization Callbacks.

.. include:: quickstart.rst

.. toctree::
   :maxdepth: 1
   :name: start

   quickstart

.. toctree::
   :maxdepth: 2
   :name: tasks
   :caption: Tasks

   tasks/nlp
   tasks/vision
   tasks/custom

.. toctree::
   :maxdepth: 2
   :name: datasets
   :caption: Datasets

   datasets/nlp
   datasets/vision
   datasets/custom

.. toctree::
   :maxdepth: 2
   :name: optimization
   :caption: Training Optimizations

   optimizations/deepspeed
   optimizations/sharded


.. toctree::
   :maxdepth: 1
   :name: api
   :caption: API Reference

   structure/conf
   structure/code
   api/lightning_transformers.core

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. This is here to make sphinx aware of the modules but not throw an error/warning
.. toctree::
   :hidden:

   readme
