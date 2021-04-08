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

To use this task, we must select a Seq2Seq Encoder/Decoder based model, such as `T5 <https://huggingface.co/transformers/model_doc/t5.html>`_ or `BART <https://huggingface.co/transformers/model_doc/bart.html>`_. Conventional models like `GPT <https://huggingface.co/transformers/model_doc/gpt.html>`_/`BERT <https://huggingface.co/transformers/model_doc/bert.html>`_ will not work as they are encoder only.

.. code-block:: bash

    python train.py task=nlp/summarization dataset=nlp/summarization/cnn_dailymail backbone.pretrained_model_name_or_path=t5-base # dataset can be swapped to xsum

.. include:: /datasets/nlp/summarization_data.rst

Summarization Inference Pipeline (experimental)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default we use the summarization pipeline, which requires an input document as text.

For Hydra to correctly parse your input argument, if your input contains any special characters you must either wrap the entire call in single quotes like `'+x="my, sentence"'` or escape special characters. See `escaped characters in unquoted values <https://hydra.cc/docs/advanced/override_grammar/basic/#escaped-characters-in-unquoted-values>`_.

.. code-block:: bash

    python predict.py task=nlp/summarization +checkpoint_path=/path/to/model.ckpt '+x="The car was racing towards the tunnel, whilst blue lights were flashing behind it. The car entered the tunnel and vanished..."'

You can also run prediction using a default HuggingFace pre-trained model:

.. code-block:: bash

   python predict.py task=nlp/summarization '+x="The car was racing towards the tunnel, whilst blue lights were flashing behind it. The car entered the tunnel and vanished..."'

Or run prediction on a specified HuggingFace pre-trained model:

.. code-block:: bash

   python predict.py task=nlp/summarization backbone.pretrained_model_name_or_path=t5-base '+x="The car was racing towards the tunnel, whilst blue lights were flashing behind it. The car entered the tunnel and vanished..."'
