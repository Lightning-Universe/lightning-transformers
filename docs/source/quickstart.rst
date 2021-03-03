Quick Start Guide
*****************

We support many NLP tasks that can be seen in :ref:`nlp-tasks`. In this example we will fine-tune BERT using the text classification task and the ``emotion`` dataset.

See ``conf/task`` and ``conf/dataset`` in the repository for more information.

Fine-tune `bert-based-cased <https://huggingface.co/bert-base-cased>`_ as default:

.. code-block:: bash

   python train.py +task=nlp/huggingface/text_classification +dataset=nlp/text_classification/emotion

Swap to `RoBERTa <https://huggingface.co/roberta-base>`_:

.. code-block:: bash

   python train.py +task=nlp/huggingface/text_classification +dataset=nlp/text_classification/emotion backbone.pretrained_model_name_or_path=roberta-base

Enable `Pytorch Lightning Native 16bit precision <https://pytorch-lightning.readthedocs.io/en/latest/amp.html#gpu-16-bit>`_:

.. code-block:: bash

   python train.py +task=nlp/huggingface/text_classification +dataset=nlp/text_classification/emotion trainer.precision=16

Swap to using RMSProp optimizer (see ``conf/optimizers/`` for all supported optimizers):

.. code-block:: bash

   python train.py +task=nlp/huggingface/text_classification +dataset=nlp/text_classification/emotion optimizer=rmsprop

Run inference once model trained (under construction):

.. code-block:: bash

   python predict.py +task=nlp/huggingface/text_classification +model=/path/to/model.ckpt +input="Classify this sentence."

   # Returns {"label_0": 0.8, "label_1": 0.2}

There are many other supported NLP tasks and datasets, see :ref:`nlp-tasks` to get started.

Fine-tuning Image GPT (under construction)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Train Image GPT ported from `Teddy Koker's implementation <https://github.com/teddykoker/image-gpt>`_ on CIFAR10.

.. code-block:: bash

   python train.py +task=vision/igpt +dataset=vision/cifar

Run inference once model trained:

.. code-block:: bash

   python predict.py +task=vision/igpt +model=/path/to/model.ckpt +input=half_image.png +output=output.png

   # Generates other half of the image, saves to output.png

There are many other supported vision tasks and datasets, see :ref:`vision-tasks` and to get started.

Trainer Options
^^^^^^^^^^^^^^^

We expose all `Pytorch Lightning Trainer <https://pytorch-lightning.readthedocs.io/en/latest/trainer.html>`_ parameters via config files. This makes it easy to configure without touching the code.

Setting maximum epochs:

.. code-block:: bash

    python train.py +task=vision/igpt +dataset=vision/cifar trainer.max_epochs=4

Using multiple GPUs:

.. code-block:: bash

    python train.py +task=vision/igpt +dataset=vision/cifar trainer.gpus=4

Using TPUs:

.. code-block:: bash

    python train.py +task=vision/igpt +dataset=vision/cifar trainer.tpu_cores=8

See the `Pytorch Lightning Trainer <https://pytorch-lightning.readthedocs.io/en/latest/trainer.html>`_  or ``conf/trainer/default`` for all parameters.