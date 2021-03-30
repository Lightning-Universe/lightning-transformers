.. _fairscale:

Sharded Training
================

Sharded Training, provided by FairScale and Lightning shards optimizer states and gradients across multiple GPUs. This means the memory overhead per GPU is lower, as each GPU only has to maintain a partition of your optimizer state and gradients.

This is particularly useful when training larger models, and reducing the memory requirement so that you can train with larger batch sizes or fit larger models into memory.
For more information see `the Sharded Training PyTorch Lightning docs <https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#sharded-training>`_.

With multiple machines, the command has to be run on all machines either manually, or using an orchestration system such as SLURM or TorchElastic. More information can be seen in the Pytorch Lightning `Computing Cluster <https://pytorch-lightning.readthedocs.io/en/latest/advanced/cluster.html#computing-cluster>`_.

Install FairScale

.. code-block:: bash

    pip install fairscale

Here we enable sharded training for the translation task.

.. code-block:: bash

    python train.py +task=nlp/translation +dataset=nlp/translation/wmt16 trainer=sharded
