.. _igpt:

Generative Pre-Training From Pixels iGPT
----------------------------------------

Train iGPT, a transformer model (GPT) trained on sequences of pixels introduced by OpenAI. The implementation is based on a port of `teddykokers' iGPT <https://github.com/teddykoker/image-gpt>`_.
Currently we only support pre-training on CIFAR10, however this can be easily extended to support your own datasets.

.. code-block:: bash

   python train.py +task=vision/igpt +dataset=vision/cifar


Pre-train a larger variant by scaling the model size.

.. code-block:: bash

   python train.py +task=vision/igpt +dataset=vision/cifar +trainer=sharded trainer.gpus=8 backbone.num_layers=5 backbone.embed_dim=8192 backbone.num_heads=16 training.batch_size=4
