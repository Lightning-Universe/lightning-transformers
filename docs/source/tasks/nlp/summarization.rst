.. _summarization:

Summarization
-------------

The Task
^^^^^^^^
The Summarization task requires the model to summarize a document into a shorter sentence.

Datasets
^^^^^^^^
Currently supports the `CNN/DailyMail <https://huggingface.co/datasets/cnn_dailymail>`_ and `XSUM <https://huggingface.co/datasets/xsum>`_ dataset or custom input text files.

In the CNN/Daily Mail dataset, this involves taking long articles and summarizing them.

.. code-block:: none

    document: "The car was racing towards the tunnel, whilst blue lights were flashing behind it. The car entered the tunnel and vanished..."

    Model answer: "Police are chasing a car entering a tunnel."

Training
^^^^^^^^

To use this task, we must select a Seq2Seq Encoder/Decoder based model, such as `T5 <https://huggingface.co/transformers/model_doc/t5.html>`_ or `BART <https://huggingface.co/transformers/model_doc/bart.html>`_. Encoder only models like `GPT <https://huggingface.co/transformers/model_doc/gpt.html>`_/`BERT <https://huggingface.co/transformers/model_doc/bert.html>`_ will not work as they are encoder only.

.. code-block:: python

    import pytorch_lightning as pl
    from transformers import AutoTokenizer

    from lightning_transformers.task.nlp.summarization import (
        SummarizationTransformer,
        XsumSummarizationDataModule,
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="t5-base")
    model = SummarizationTransformer(
        pretrained_model_name_or_path="t5-base",
        use_stemmer=True,
        val_target_max_length=142,
        num_beams=None,
        compute_generate_metrics=True,
    )
    dm = XsumSummarizationDataModule(
        batch_size=1,
        max_source_length=128,
        max_target_length=128,
        tokenizer=tokenizer,
    )
    trainer = pl.Trainer(accelerator="auto", devices=1, max_epochs=1)

    trainer.fit(model, dm)


.. include:: /datasets/nlp/summarization_data.rst

Summarization Inference Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default we use the summarization pipeline, which requires an input document as text.

.. code-block:: python

    from transformers import AutoTokenizer
    from lightning_transformers.task.nlp.summarization import SummarizationTransformer

    model = SummarizationTransformer(
        pretrained_model_name_or_path="patrickvonplaten/t5-tiny-random",
        tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path="patrickvonplaten/t5-tiny-random"),
    )

    model.hf_predict(
        "The results found significant improvements over all tasks evaluated",
        min_length=2,
        max_length=12,
    )
