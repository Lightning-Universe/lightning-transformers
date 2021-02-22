.. _deepspeed:

DeepSpeed
=========

DeepSpeed is a Deep Learning Optimization library, offering many techniques to reduce memory/computation footprint.
This allows us to train large transformer using less compute. For more details, see <TODO lightning DeepSpeed docs>.

ZeRO-Offload
------------

ZeRO-Offload enables large model training by being efficient with memory usage. ZeRO-Offload additionally
leverages the host CPU to execute the optimizer.

We provide default trainer configurations to enable ZeRO-Offload:

.. code-block:: bash

   python train.py +task=vision/igpt +dataset=vision/cifar trainer=zero_offload

With multiple machines `(Command has to be run on all machines either manually, or using an orchestration system such as SLURM or TorchElastic)`:

.. code-block:: bash

   deepspeed --num_nodes 2 --num_gpus 1 train.py +task=vision/igpt +dataset=vision/cifar trainer=zero_offload

To see the configuration settings see ``conf/trainer/zero_offload.yaml``. Within the config file, you can modify the ZeRO parameters as described in the `DeepSpeed ZeRO docs <https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training>`_:

.. code-block:: yaml

    deepspeed_config:
      optimizer: ${optimizer}
      scheduler: ${scheduler}
      zero_optimization: # Add your custom parameters here
        stage: 2
        cpu_offload: True
        contiguous_gradients: True
        overlap_comm: True
