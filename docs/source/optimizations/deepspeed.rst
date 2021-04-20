.. _deepspeed:

DeepSpeed
=========

`DeepSpeed <https://github.com/microsoft/DeepSpeed>`__ is a deep learning training optimization library, providing the means to train massive billion parameter models at scale.
This allows us to train large transformer models optimizing for compute. For more details, see `the DeepSpeed PyTorch Lightning docs <https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#deepspeed>`__.

With multiple machines, the command has to be run on all machines either manually, or using an orchestration system such as SLURM or TorchElastic. More information can be seen in the Pytorch Lightning `Computing Cluster <https://pytorch-lightning.readthedocs.io/en/latest/advanced/cluster.html#computing-cluster>`_.

DeepSpeed ZeRO Stage 2
----------------------

We provide out of the box configs to use the DeepSpeed plugin. Below is an example of how you can swap to the default trainer config for DeepSpeed when using the translation task.

.. code-block:: python

   python train.py task=nlp/translation dataset=nlp/translation/wmt16 trainer=deepspeed

All options can be found in `conf/trainer/plugins/deepspeed.yaml <https://github.com/PyTorchLightning/lightning-transformers/blob/master/conf/trainer/plugins/deepspeed.yaml>`__. We suggest referring to `the DeepSpeed PyTorch Lightning docs <https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#deepspeed>`__ for more information on the parameters.

DeepSpeed ZeRO Stage 2 Offload
------------------------------

ZeRO-Offload enables large model training by being efficient with memory usage. ZeRO-Offload leverages the host CPU to execute the optimizer.

We provide default trainer configurations to enable ZeRO-Offload:

.. code-block:: python

   python train.py task=nlp/translation dataset=nlp/translation/wmt16 trainer=deepspeed_offload

To see the configuration settings see `conf/trainer/plugins/deepspeed_offload.yaml <https://github.com/PyTorchLightning/lightning-transformers/blob/master/conf/trainer/plugins/deepspeed_offload.yaml>`__  for the parameters. Within the config file, you can modify the ZeRO parameters as described in `the DeepSpeed PyTorch Lightning docs <https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#deepspeed>`__.

.. code-block:: yaml

    _target_: pytorch_lightning.plugins.DeepSpeedPlugin
    stage: 2
    cpu_offload: True
