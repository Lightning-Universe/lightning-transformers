Translation
-----------
The Translation task fine-tunes the model to translate text from one language to another. Currently supports the `WMT16 <https://huggingface.co/datasets/wmt16>`_ dataset or custom input text files.

.. code-block:: none

    Input Text: <TODO>
    Model Output: <TODO>

To use this task, we must select a Seq2Seq Encoder/Decoder based model, such as multi-lingual T5 or BART. Conventional models like GPT/BERT will not work as they are encoder only.
In addition, we also need a tokenizer that has been created on multi-lingual text. This is true for `mt5 <https://huggingface.co/google/mt5-base>`_ and `mbart <https://huggingface.co/facebook/mbart-large-cc25>`_.

.. code-block:: bash

    python train.py +task=nlp/huggingface/translation +dataset=nlp/translation/wmt16 backbone.pretrained_model_name_or_path=google/mt5-base

By default we fine-tune to translate from English to Romanian. This can be changed by specifying the source/target languages (see more in ``conf/dataset/nlp/translation/wmt16.yaml``):

.. code-block:: bash

    python train.py +task=nlp/huggingface/translation +dataset=nlp/translation/wmt16 backbone.pretrained_model_name_or_path=google/mt5-base dataset.dataset_config_name=de-en dataset.src_lang=de dataset.tgt_lang=en


Translation Using Custom Files (under construction)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use custom text files, the files should contain data in the same format as the `WMT16 format <https://huggingface.co/datasets/wmt16#data-instances>`_:

.. code-block:: json

    {
       "answers":{
          "answer_start":[
             1
          ],
          "text":[
             "This is a test text"
          ]
       },
       "context":"This is a test context.",
       "id":"1",
       "question":"Is this a test?",
       "title":"train test"
    }

We override the dataset files, allowing us to still use the data transforms defined with this dataset.

.. code-block:: bash

    python train.py +task=nlp/huggingface/translation +dataset=nlp/translation/wmt16 backbone.pretrained_model_name_or_path=google/mt5-base dataset.train_file=train.txt dataset.validation_file=valid.txt

Translation Inference Pipeline (under construction)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default we use the translation pipeline, which requires a source text string.

.. code-block:: bash

    python predict.py +task=nlp/huggingface/translation +model=/path/to/model.ckpt input="The ground is black, the sky is blue and the car is red."