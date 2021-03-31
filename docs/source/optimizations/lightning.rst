.. _lightning:

Lightning Trainer
=================

The Lightning Trainer offers many advanced features and training optimization without changing your models.

Any Trainer flag can be used in Lightning transformers by passing ``trainer.FLAG=VALUE`` to ``train.py``.

Max epochs
""""""""""

.. code-block:: python

   python train.py +task=nlp/text_classification +dataset=nlp/text_classification/emotion trainer.max_epochs=4

16-bit precision
""""""""""""""""
.. code-block:: python

   python train.py +task=nlp/text_classification +dataset=nlp/text_classification/emotion trainer.precision=16

Multi GPU
"""""""""
.. code-block:: python

    python train.py +task=nlp/translation +dataset=nlp/translation/wmt16 trainer.gpus=4

TPUs
""""
.. code-block:: python

    python train.py +task=nlp/translation +dataset=nlp/translation/wmt16 trainer.tpu_cores=8

See the `Pytorch Lightning Trainer <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_  or `conf/trainer/default <https://github.com/PyTorchLightning/lightning-transformers/blob/master/conf/trainer/default.yaml>`_ for all the Lightning Trainer options.
