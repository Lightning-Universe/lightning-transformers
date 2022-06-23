.. _sparseml:

SparseML
========

`SparseML <https://github.com/neuralmagic/sparseml>`__ provides GPU-class performance on CPUs through sparsification, pruning, and quantization.
For more details, see `SparseML docs <https://docs.neuralmagic.com/sparseml/>`__.

With multiple machines, the command has to be run on all machines either manually, or using an orchestration system such as SLURM or TorchElastic. More information can be seen in the Pytorch Lightning `Computing Cluster <https://pytorch-lightning.readthedocs.io/en/latest/advanced/cluster.html#computing-cluster>`_.

We provide out of the box configs to use SparseML. Just pass the SparseML Callback when training.

.. code-block:: python

    import pytorch_lightning as pl
    from lightning_transformers.callbacks import TransformerSparseMLCallback

    pl.Trainer(
        callbacks=TransformerSparseMLCallback(
            output_dir="/content/MODELS",
            recipe_path="/content/recipe.yaml"
        )
    )

These commands are only useful when a recipe has already been created. Example recipes can be found `here <https://github.com/neuralmagic/sparseml/tree/main/integrations/huggingface-transformers/recipes>`__.

After training, this will leave two ONNX models in the trainer.callbacks.output_dir folder: small_model.onnx and model.onnx. small_model.onnx is excellent for demos. For reliable inference, it is recommended to optimize model.onnx with your compression algorithm.
