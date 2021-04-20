.. _quickstart:

Quick Start
===========


.. include:: installation.rst


Using Lightning-Transformers
****************************

Lightning Transformers has a collections of tasks for common NLP problems such as :ref:`language_modeling`, :ref:`translation` and more. To use, simply:

1. Pick a task to train (passed to ``train.py`` as ``task=``)

2. Pick a dataset (passed to ``train.py`` as ``dataset=``)

3. Customize the backbone, optimizer, or any component within the config

4. Add any `Lightning supported parameters and optimizations <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_

.. code-block:: python

   python train.py \
        task=<TASK> \
        dataset=<DATASET> \
        backbone.pretrained_model_name_or_path=<BACKBONE> # Optionally change the HF backbone model
        optimizer=<OPTIMIZER> # Optionally specify optimizer (Default AdamW)
        trainer.<ANY_TRAINER_FLAGS> # Optionally specify Lightning trainer arguments

Finetuning
----------

In this example we will finetune the :ref:`text_classification` task on the `emotion <https://huggingface.co/datasets/emotion>`_ dataset.

To fine-tune using the Text Classification default `bert-based-cased <https://huggingface.co/bert-base-cased>`_ model, use the following:

.. code-block:: python

   python train.py task=nlp/text_classification dataset=nlp/text_classification/emotion

Change the Backbone
-------------------

Swap to the `RoBERTa <https://huggingface.co/roberta-base>`_ model:

.. code-block:: python

   python train.py task=nlp/text_classification dataset=nlp/text_classification/emotion backbone.pretrained_model_name_or_path=roberta-base

Change the Optimizer
--------------------

Swap to using RMSProp optimizer (see `conf/optimizer/ <https://github.com/PyTorchLightning/lightning-transformers/tree/master/conf/optimizer>`_ for all supported optimizers):

.. code-block:: python

   python train.py task=nlp/text_classification dataset=nlp/text_classification/emotion optimizer=rmsprop

For more info on how to override configuration, see :ref:`conf`.

Lightning Trainer Options
--------------------------

We expose all `Pytorch Lightning Trainer <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ parameters via the ``trainer`` config. This makes it easy to configure the Lightning Trainer without touching the code.

Below we enable `Pytorch Lightning Native 16bit precision <https://pytorch-lightning.readthedocs.io/en/latest/amp.html#gpu-16-bit>`_ by setting the parameter via the CLI:

.. code-block:: python

   python train.py task=nlp/text_classification dataset=nlp/text_classification/emotion trainer.precision=16


Setting the maximum epochs:

.. code-block:: python

    python train.py task=nlp/translation dataset=nlp/translation/wmt16 trainer.max_epochs=4

Using multiple GPUs:

.. code-block:: python

    python train.py task=nlp/translation dataset=nlp/translation/wmt16 trainer.gpus=4

Using TPUs:

.. code-block:: python

    python train.py task=nlp/translation dataset=nlp/translation/wmt16 trainer.tpu_cores=8

See the `Pytorch Lightning Trainer <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_  or `conf/trainer/default <https://github.com/PyTorchLightning/lightning-transformers/blob/master/conf/trainer/default.yaml>`_ for all parameters.

Inference
---------

Run inference once model trained (experimental):

.. code-block:: python

   python predict.py task=nlp/text_classification +checkpoint_path=/path/to/model.ckpt +x="Classify this sentence."

   # Returns [{'label': 'LABEL_0', 'score': 0.545...}]

You can also run prediction using the default HuggingFace pre-trained model:

.. code-block:: python

   python predict.py task=nlp/text_classification +x="Classify this sentence."

Or run prediction on a specified HuggingFace pre-trained model:

.. code-block:: python

   python predict.py task=nlp/text_classification backbone.pretrained_model_name_or_path=bert-base-cased +x="Classify this sentence."
