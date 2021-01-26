.. _deepspeed:

DeepSpeed
=========

TODO explanation of DeepSpeed and what ZeRO Offload is

Enable `DeepSpeed ZeRO Offload <https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html#deepspeed>`_ (under construction):

.. code-block:: bash

   python train.py +task=vision/igpt +dataset=vision/cifar trainer=zero_offload

TODO Show what zero_offload enables, and how to modify parameters with link to DeepSpeed docs.

With multiple machines `(Command has to be run on all machines either manually, or using an orchestration system such as SLURM or TorchElastic)`:

.. code-block:: bash

   deepspeed --num_nodes 2 --num_gpus 1 train.py +task=vision/igpt +dataset=vision/cifar trainer=zero_offload
