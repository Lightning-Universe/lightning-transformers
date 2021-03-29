Language Modeling Using Your Own Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use custom text files, the files should contain the raw data you want to train and validate on.

During data pre-processing the text is flattened, and the model is trained and validated on context windows (block size) made from the input text. We override the dataset files, allowing us to still use the data transforms defined with the base datamodule.

Below we have defined a csv file to use as our input data.

.. code-block::

    text,
    this is the first sentence,
    this is the second sentence,


.. code-block:: bash

    python train.py +task=nlp/language_modeling dataset.train_file=train.csv dataset.validation_file=valid.csv

Language Modeling using Custom Data Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In some cases you would like to modify the data processing logic before training, validation and testing.

Ultimately to create your own custom data processing the flow is like this:

1. Extend the ``LanguageModelingTransformerDataModule`` base class, Override hooks with your own logic
2. (Optional) Keep file in the specific task directory
3. Add a hydra config object to use your new dataset

Extend the ``LanguageModelingTransformerDataModule`` base class
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The base data module can be used to modify this code, and follows a simple pattern. Internally the dataset is loaded via HuggingFace Datasets, which returns an Arrow Dataset. This data format is easy to transform and modify using map functions, which you'll see within the class.

.. code-block:: python

    class LanguageModelingTransformerDataModule(HFTransformerDataModule):
    cfg: LanguageModelingDataConfig # The config options passed to the constructor

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        # `process_data` converting the dataset into features.
        # The dataset is pre-loaded using `load_dataset`.
        ...
        return dataset

    @staticmethod
    def tokenize_function(
        examples,
        tokenizer: Union[Tokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast],
        text_column_name: str = None,
    ):
        # tokenizes the data in a specific column using the AutoTokenizer,
        # called by `process_data`
        return tokenizer(examples[text_column_name])

    @staticmethod
    def convert_to_features(examples, block_size: int = None):
        # `process_data` calls this function to convert samples in the dataset into features
        ...

    @property
    def collate_fn(self) -> Callable:
        # `Describes how to collate the samples for the batch given to the model`
        return default_data_collator



Extend ``LanguageModelingTransformerDataModule``, like this.

.. code-block:: python

    from lightning_transformers.task.nlp.language_modeling import LanguageModelingTransformerDataModule

    class MyLanguageModelingDataModule(LanguageModelingTransformerDataModule):
        ...

Make any changes you'd like to the dataset processing via the hooks.

To see a full example see ``examples/custom/language_modeling/custom_dataset.py``

(Optional) Keep file in the specific task directory
"""""""""""""""""""""""""""""""""""""""""""""""""""

This makes tracking of files easier. Our example is stored in ``examples/`` however in reality we would store our DataModule in ``lightning_transformers/task/nlp/language_modeling/custom_dataset.py``.

Add a hydra config object to use your new dataset
"""""""""""""""""""""""""""""""""""""""""""""""""

Finally to use the Hydra CLI and configs, we would add our own custom yaml file containing the necessary code to run using our dataset.

We create a file at ``conf/datasets/nlp/language_modeling/my_dataset.yaml`` containing the below config.

.. code-block:: yaml

    # @package dataset
    defaults:
      - nlp/hf_default # Use the defaults from the default config found at `conf/dataset/nlp/hf_default.yaml`
    _target_: lightning_transformers.task.nlp.language_modeling.my_dataset.MyLanguageModelingDataModule # path to the class we'd like to instantiate
    cfg:
      block_size: 512 # any parameters you'd like from the inherited config object.

With this in place you can now train using either HuggingFace Datasets or your own custom files.

.. code-block:: bash

    python train.py +task=nlp/language_modeling +dataset=nlp/language_modeling/my_dataset dataset.train_file=train.csv dataset.validation_file=valid.csv
