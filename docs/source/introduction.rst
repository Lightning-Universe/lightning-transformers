Lightning Transformers provides capabilities for high performance research using SOTA Transformers.

Why Lightning Transformers?
***************************

* **Powered by PyTorch Lightning** - Leverage everything that `PyTorch Lightning <https://github.com/PyTorchLightning/pytorch-lightning>`_ has to offer, allowing you to use Lightning provided and custom Callbacks, Loggers, Accelerators and high performance scaling with minimal changes.
* **Backed by HuggingFace Transformers** - Train using `HuggingFace Transformer <https://github.com/huggingface/transformers>`_ models and datasets, across the expansive library spanning multiple modalities and tasks within NLP/Audio and Vision.
* **Transformer Task Abstraction for Rapid Research & Experimentation** - Built from the ground up to be task agnostic, the library supports creating transformer tasks across all modalities with little friction.
* **Powerful config composition backed by Hydra** - Leverage the config structure to swap out models, optimizers, schedulers task and many more configurations without touching the code.
* **Seamless Memory and Speed Optimizations** - We provide seamless integration to enable training optimizations, such as `DeepSpeed ZeRO <https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html#deepspeed>`_ or `FairScale Sharded Training <https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html#sharded-training>`_ with no code changes.

For a list of currently supported tasks, see :ref:`tasks`.
