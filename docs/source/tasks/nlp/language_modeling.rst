.. _language_modeling:

Language Modeling
-----------------

The Task
^^^^^^^^
Causal Language Modeling is the vanilla autoregressive pre-training method common to most language models such as `GPT-3 <https://arxiv.org/abs/2005.14165>`_ or `CTRL <https://arxiv.org/abs/1909.05858>`_
(Excluding BERT-like models, which were pre-trained using the Masked Language Modeling training method).

During training, we minimize the maximum likelihood during training across spans of text data (usually in some context window/block size).
The model is able to attend to the left context (left of the mask).
When trained on large quantities of text data, this gives us strong language models such as GPT-3 to use for downstream tasks.

Datasets
^^^^^^^^
Currently supports the `wikitext2 <https://huggingface.co/datasets/wikitext>`_ dataset, or custom input files.
Since this task is usually the pre-training task for Transformers, it can be used to train new language models from scratch or to fine-tune a language model onto your own unlabeled text data.

Usage
^^^^^

Language Models pre-trained or fine-tuned to the Causal Language Modeling task can then be used in generative predictions.

.. code-block:: python

    import pytorch_lightning as pl
    from transformers import AutoTokenizer

    from lightning_transformers.task.nlp.language_modeling import (
        LanguageModelingDataModule,
        LanguageModelingTransformer,
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="gpt2")
    model = LanguageModelingTransformer(pretrained_model_name_or_path="gpt2")
    dm = LanguageModelingDataModule(
        batch_size=1,
        dataset_name="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        tokenizer=tokenizer,
    )
    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=1)

    trainer.fit(model, dm)


We report the Cross Entropy Loss for validation.

.. include:: /datasets/nlp/language_modeling_data.rst

Language Modeling Inference Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default we use the text generation pipeline, which requires a conditional input string and generates an output string.

.. code-block:: python

    from transformers import AutoTokenizer
    from lightning_transformers.task.nlp.language_modeling import LanguageModelingTransformer

    model = LanguageModelingTransformer(
            pretrained_model_name_or_path="prajjwal1/bert-tiny",
            tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path="prajjwal1/bert-tiny"),
        )
    model.hf_predict("The house:")
