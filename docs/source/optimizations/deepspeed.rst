.. _deepspeed:

DeepSpeed
=========

DeepSpeed is a Deep Learning Optimization library, offering many techniques to reduce memory/computation footprint.
This allows us to train large transformer using less compute. For more details, see `the DeepSpeed PyTorch Lightning docs <https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#deepspeed>`_.

ZeRO-Offload
------------

ZeRO-Offload enables large model training by being efficient with memory usage. ZeRO-Offload additionally
leverages the host CPU to execute the optimizer.

We provide default trainer configurations to enable ZeRO-Offload:

.. code-block:: bash

   python train.py +task=nlp/translation +dataset=nlp/translation/wmt16 trainer=zero_offload

With multiple machines, the above command has to be run on all machines either manually, or using an orchestration system such as SLURM or TorchElastic. More information can be seen in the Pytorch Lightning `Computing Cluster <https://pytorch-lightning.readthedocs.io/en/latest/advanced/cluster.html#computing-cluster>`_.

To see the configuration settings see ``conf/trainer/zero_offload.yaml`` and ``conf/trainer/plugins/zero_offload.yaml`` for the parameters. Within the config file, you can modify the ZeRO parameters as described in `the DeepSpeed PyTorch Lightning docs <https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#deepspeed>`_.

.. code-block:: yaml

    deepspeed_config:
      optimizer: ${optimizer}
      scheduler: ${scheduler}
      zero_optimization: # Add your custom parameters here
        stage: 2
        cpu_offload: True
        contiguous_gradients: True
        overlap_comm: True
