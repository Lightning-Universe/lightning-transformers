.. _question_answering:

Question Answering
------------------

The Task
^^^^^^^^
The Question Answering task requires the model to determine the start and end of a span within the given context, that answers a given question.
This allows the model to pre-condition on contextual information to determine an answer.

Use this task when you would like to fine-tune onto data where an answer can be extracted from context information.
Since this is an extraction task, you can rely on most Transformer models as your backbone.

Datasets
^^^^^^^^
Currently supports the `SQuAD <https://huggingface.co/datasets/squad>`_ dataset or custom input text files.

.. code-block:: none

    Context: The ground is black, the sky is blue and the car is red.
    Question: What color is the sky?

    Model answer: {"answer": "the sky is blue", "start": 21, "end": 35}

Training
^^^^^^^^

.. code-block:: python
    import pytorch_lightning as pl
    from transformers import AutoTokenizer

    from lightning_transformers.task.nlp.question_answering import (
        QuestionAnsweringDataConfig,
        QuestionAnsweringTransformer,
        SquadDataModule,
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-uncased")
    model = QuestionAnsweringTransformer(pretrained_model_name_or_path="bert-base-uncased")
    dm = SquadDataModule(
        cfg=QuestionAnsweringDataConfig(
            batch_size=1,
            dataset_name="squad",
            dataset_config_name="plain_text",
            max_length=384,
            version_2_with_negative=False,
            null_score_diff_threshold=0.0,
            doc_stride=128,
            n_best_size=20,
            max_answer_length=30,
        ),
        tokenizer=tokenizer,
    )
    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=1)

    trainer.fit(model, dm)

.. include:: /datasets/nlp/question_answering_data.rst

Question Answering Inference Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default we use the question answering pipeline, which requires a context and a question as input.

.. code-block:: python

    from transformers import AutoTokenizer
    from lightning_transformers.task.nlp.question_answering import QuestionAnsweringTransformer

    model = QuestionAnsweringTransformer(
        pretrained_model_name_or_path="sshleifer/tiny-gpt2",
        tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path="sshleifer/tiny-gpt2"),
    )
    model.hf_predict(dict(context="Lightning is great", question="What is great?"))
