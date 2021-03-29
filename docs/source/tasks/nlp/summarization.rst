Summarization
-------------
The Summarization task requires the model to summarize a document into a shorter sentence.
Currently supports the `CNN/DailyMail <https://huggingface.co/datasets/cnn_dailymail>`_ and `XSUM <https://huggingface.co/datasets/xsum>`_ dataset or custom input text files.

In the CNN/Daily Mail dataset, this involves taking long articles and summarizing them.

.. code-block:: none

    document: "The car was racing towards the tunnel, whilst blue lights were flashing behind it. The car entered the tunnel and vanished..."

    Model answer: "Police are chasing a car entering a tunnel."

To use this task, we must select a Seq2Seq Encoder/Decoder based model, such as T5 or BART. Conventional models like GPT/BERT will not work as they are encoder only.

.. code-block:: bash

    python train.py +task=nlp/summarization +dataset=nlp/summarization/cnn_dailymail backbone.pretrained_model_name_or_path=t5-base # dataset can be swapped to xsum

.. include:: /datasets/nlp/summarization_data.rst

Summarization Inference Pipeline (under construction)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default we use the summarization pipeline, which requires an input document as text.

.. code-block:: bash

    python predict.py +task=nlp/question_Answering +model=/path/to/model.ckpt input="The car was racing towards the tunnel, whilst blue lights were flashing behind it. The car entered the tunnel and vanished..."
