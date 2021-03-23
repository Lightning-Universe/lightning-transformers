.. _fairscale:

FairScale Sharded Training
==========================

TODO Description of sharded training, installing fairscale

Enable `Sharded Training <https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html#sharded-training>`_:

.. code-block:: bash

   python train.py +task=nlp/translation +dataset=nlp/translation/wmt16 trainer=sharded

TODO: Description of what the sharded.yaml file enables
