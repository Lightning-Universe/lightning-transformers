.. _translation:

Translation
-----------

The Task
^^^^^^^^
The Translation task fine-tunes the model to translate text from one language to another.


Datasets
^^^^^^^^
Currently supports the `WMT16 <https://huggingface.co/datasets/wmt16>`_ dataset or custom input text files.

.. code-block:: none

    Input Text (English): "The ground is black, the sky is blue and the car is red."
    Model Output (German): "Der Boden ist schwarz, der Himmel ist blau und das Auto ist rot."

Training
^^^^^^^^

To use this task, select a Seq2Seq Encoder/Decoder based model, such as multi-lingual `T5 <https://huggingface.co/transformers/model_doc/t5.html>`_ or `BART <https://huggingface.co/transformers/model_doc/bart.html>`_. Conventional models like `GPT <https://huggingface.co/transformers/model_doc/gpt.html>`_/`BERT <https://huggingface.co/transformers/model_doc/bert.html>`_ will not work as they are encoder only.
In addition, you also need a tokenizer that has been created on multi-lingual text. This is true for `mt5 <https://huggingface.co/google/mt5-base>`_ and `mbart <https://huggingface.co/facebook/mbart-large-cc25>`_.

.. code-block:: python

    import pytorch_lightning as pl
    from transformers import AutoTokenizer

    from lightning_transformers.task.nlp.translation import (
        TranslationTransformer,
        WMT16TranslationDataModule,
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="t5-base")
    model = TranslationTransformer(
        pretrained_model_name_or_path="t5-base",
        n_gram=4,
        smooth=False,
        val_target_max_length=142,
        num_beams=None,
        compute_generate_metrics=True,
    )
    dm = WMT16TranslationDataModule(
        # WMT translation datasets: ['cs-en', 'de-en', 'fi-en', 'ro-en', 'ru-en', 'tr-en']
        dataset_config_name="ro-en",
        source_language="en",
        target_language="ro",
        max_source_length=128,
        max_target_length=128,
        tokenizer=tokenizer,
    )
    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=1)

    trainer.fit(model, dm)

.. include:: /datasets/nlp/translation_data.rst

Translation Inference Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default we use the translation pipeline, which requires a source text string.

.. code-block:: python

    from transformers import AutoTokenizer
    from lightning_transformers.task.nlp.translation import TranslationTransformer

    model = TranslationTransformer(
        pretrained_model_name_or_path="patrickvonplaten/t5-tiny-random",
        tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path="patrickvonplaten/t5-tiny-random"),
    )
    model.hf_predict("¡Hola Sean!")
