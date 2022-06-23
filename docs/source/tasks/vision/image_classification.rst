.. _image_classification:

Image Classification
--------------------

The Task
^^^^^^^^
Image classification is the task of classifying an image to a label or task.


Datasets
^^^^^^^^
Currently supports the `beans <https://huggingface.co/datasets/beans>`_ dataset, or custom input files.

Usage
^^^^^

Language Models pre-trained or fine-tuned to the Causal Language Modeling task can then be used in generative predictions.

.. code-block:: python

    import pytorch_lightning as pl
    from transformers import AutoFeatureExtractor

    from lightning_transformers.task.vision.image_classification import (
        ImageClassificationDataModule,
        ImageClassificationTransformer,
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path="nateraw/vit-base-beans")
    dm = ImageClassificationDataModule(
        batch_size=8,
        dataset_name="beans",
        num_workers=8,
        feature_extractor=feature_extractor,
    )
    model = ImageClassificationTransformer(pretrained_model_name_or_path="nateraw/vit-base-beans", num_labels=dm.num_classes)

    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=5)
    trainer.fit(model, dm)



We report the Cross Entropy Loss for validation.

Image Classification Inference Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from transformers import AutoTokenizer
    from lightning_transformers.task.vision.image_classification import ImageClassificationTransformer

    model = ImageClassificationTransformer(
        pretrained_model_name_or_path="nateraw/vit-base-beans",
        tokenizer=AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path="nateraw/vit-base-beans"),
    )
    # predict on the logo
    model.hf_predict(
        "https://github.com/PyTorchLightning/lightning-transformers/blob/master/"
        "docs/source/_static/images/logo.png?raw=true"
    )
