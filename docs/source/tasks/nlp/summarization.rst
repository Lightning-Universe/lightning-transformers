Summarization
-------------
Fine-tune Transformers using the Summarization Task.
Currently supports the `CNN/DailyMail <https://huggingface.co/datasets/cnn_dailymail>`_ and `XSUM <https://huggingface.co/datasets/xsum>`_ dataset or custom input text files.

The Summarization task requires the model to summarize a document into a shorter sentence. In the CNN/Daily Mail dataset, this involves taking long articles and summarizing them.

.. code-block:: none

    document: "The car was racing towards the tunnel, whilst blue lights were flashing behind it. The car entered the tunnel and vanished..."

    Model answer: "Police are chasing a car entering a tunnel."

To use this task, we must select a Seq2Seq Encoder/Decoder based model, such as T5 or BART. Conventional models like GPT/BERT will not work as they are encoder only.

.. code-block:: bash

    python train.py +task=nlp/huggingface/summarization +dataset=nlp/summarization/cnn_dailymail backbone.pretrained_model_name_or_path=t5-base # dataset can be swapped to xsum

Summarization Using Custom Files (under construction)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use custom text files, the files should contain new line delimited json objects within the text files, in the same format as `XSUM dataset <https://huggingface.co/datasets/xsum#data-instances>`_:

.. code-block:: json

    {
        "document": "some-body",
        "id": "29750031",
        "summary": "some-sentence"
    }

We override the dataset files, allowing us to still use the data transforms defined with this dataset.

.. code-block:: bash

    python train.py +task=nlp/huggingface/summarization +dataset=nlp/summarization/xsum backbone.pretrained_model_name_or_path=t5-base dataset.train_file=train.txt dataset.validation_file=valid.txt

Summarization Inference Pipeline (under construction)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default we use the summarization pipeline, which requires an input document as text.

.. code-block:: bash

    python predict.py +task=nlp/huggingface/question_Answering +model=/path/to/model.ckpt input="The car was racing towards the tunnel, whilst blue lights were flashing behind it. The car entered the tunnel and vanished..."
