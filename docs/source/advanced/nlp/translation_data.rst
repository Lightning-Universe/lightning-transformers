Translation using Custom Data Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below we show an example of how overriding data processing logic, by adding a prefix to the source language sample in translation. Check :doc:`/tasks/nlp/translation` for more information around the task.

Ultimately to create your own custom data processing the flow is like this:

1. Extend the ``TranslationDataModule`` base class, Override hooks with your own logic
2. (Optional) Keep file in the specific task directory
3. Add a hydra config object to use your new dataset

1. Extend the ``TranslationDataModule`` base class
""""""""""""""""""""""""""""""""""""""""""""""""""

The base data module can be used to modify this code, and follows a simple pattern. Internally the dataset is loaded via HuggingFace Datasets, which returns an `Apache Arrow Parquet <https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetDataset.html>`_ Dataset. This data format is easy to transform and modify using map functions, which you'll see within the class.

.. code-block:: python

    class TranslationDataModule(Seq2SeqDataModule):

        def __init__(self, cfg: TranslationDataConfig = TranslationDataConfig()):
            super().__init__(cfg=cfg)

        @property
        def source_target_column_names(self) -> Tuple[str, str]:
            return self.cfg.source_language, self.cfg.target_language

    ...

    class Seq2SeqDataModule(HFDataModule):

        def __init__(self, cfg: Seq2SeqDataConfig = Seq2SeqDataConfig()):
            super().__init__(cfg=cfg)

        def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
            # `process_data` converting the dataset into features.
            # The dataset is pre-loaded using `load_dataset`.
            ...
            return dataset

        @property
        def source_target_column_names(self) -> Tuple[str, str]:
            return 'source', 'target'

        @staticmethod
        def convert_to_features(examples, block_size: int = None):
            # `process_data` calls this function to convert samples in the dataset into features
            ...

        @property
        def collate_fn(self) -> Callable:
            # `Describes how to collate the samples for the batch given to the model`
            return default_data_collator


Extend ``TranslationDataModule``, like this.

.. code-block:: python

    from lightning_transformers.task.nlp.translation import TranslationDataModule

    class MyTranslationDataModule(TranslationDataModule):
        ...

Make any changes you'd like to the dataset processing via the hooks.

To see the full example, see ``examples/custom/dataset/translation/custom_dataset.py``

2. (Optional) Keep file in the specific task directory
""""""""""""""""""""""""""""""""""""""""""""""""""""""

This makes tracking of files easier. Our example is stored in ``examples/`` however in reality we would store our DataModule in ``lightning_transformers/task/nlp/translation/datasets/custom_dataset.py``.

3. Add a hydra config object to use your new dataset
""""""""""""""""""""""""""""""""""""""""""""""""""""

Finally to use the Hydra CLI and configs, we would add our own custom yaml file containing the necessary code to run using our dataset.

We create a file at ``conf/datasets/nlp/translation/my_dataset.yaml`` containing the below config.

.. code-block:: yaml

    # @package dataset
    defaults:
      - nlp/default # Use the defaults from the default config found at `conf/dataset/nlp/default.yaml`
    _target_: examples.custom_translation.dataset.MyTranslationDataModule # path to the class we'd like to instantiate
    cfg:
      max_source_length: 128 # any parameters you'd like from the inherited config object.

With this in place you can now train using either HuggingFace Datasets or your own custom files.

.. code-block:: bash

    python train.py +task=nlp/translation +dataset=nlp/translation/my_dataset dataset.cfg.train_file=train.csv dataset.cfg.validation_file=valid.csv
