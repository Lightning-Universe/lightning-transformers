.. _new-task:

Building New Tasks
==================

Below we demonstrate how to build a custom task within Lightning Transformers.
When building a custom task in Lightning Transformers, there are three key pieces.

* Implement the Dataset
* Implement the Backbone
* Implement the Task

In some cases you'll need to implement all three, and in most cases you will need to implement a subset of them. There is also interconnection between
sections; for example when your model requires knowing how many classes are within your dataset. We cover all these details below.

Since datasets and backbones are mostly specific to tasks, the code is organized into tasks. You can create your own folder and add all the necessary logic within a new task folder for organization.

.. code-block:: bash

    lightning_transformers/
    ┣ task/
        ┣ nlp/
        ┣    ...
        ┣ vision/
             igpt/
                ...

In addition, make sure you are familiar with the config structure as this will be how you can modify your parameters via the cmdline. See :ref:`conf`.

The Dataset
-----------

In our example we build the CIFAR10 dataset for iGPT, more details at :ref:`igpt`.

The base class for Lightning Transformer datasets is the ``TransformerDataModule`` class, which is a thin layer on top of the LightningDataModule class, exposing all the Lightning Data Hooks as standard. See `LightningDataModule <https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html>`_ for more details.
Additional, the base class provides a few helper functions, plus allows us to define any data specific arguments we'd like the task to be aware of at initialize time.

In some cases you do not need to define a module, and could use a pre-set LightningDataModule out the box.

Here is a simplified example of the iGPT DataModule.

.. code-block:: python

    class ImageGPTDataModule(TransformerDataModule):
        cfg: ImageGPTDataConfig

        def __init__(self, cfg: ImageGPTDataConfig):
            ...

        def prepare_data(self, *args: Any, **kwargs: Any) -> None:
            """
            Called on ONE process (like 1 GPU/ 1 TPU). Useful to download datasets.
            """
            ...

        def setup(self, stage: Optional[str] = None):
            """
            Called on every process to setup the datamodule
            """
            ...

        @property
        def model_data_args(self) -> Dict:
            """
            Override to provide the model with additional args.
            This is useful to provide the number of classes/pixels to the model or any other data specific args
            Returns: Dict of args
            """
            return {}

We also define a dataclass to define the input config to the class. This makes it easier to pass options around.

.. code-block:: python
    @dataclass
    class ImageGPTDataConfig(TransformerDataConfig):
        dataset: str = "cifar10"
        data_dir: Optional[Union[str, Path]] = None

See ``lightning_transformers/task/vision/igpt/data.py`` for the full implementation.

At runtime we also need to define a config. You can find the config for iGPT CIFAR10 at ``conf/datasets/vision/igpt/cifar``.

.. code-block:: yaml

    # @package dataset
    defaults:
      - default
    _target_: lightning_transformers.task.vision.igpt.data.ImageGPTDataModule
    cfg:
      dataset: cifar10
      data_dir: ./


Here we define the ``_target_`` class we'll like to instantiate (our DataModule that we just implemented) as well as any specific arguments.
We also inherit from a default config which can be found in ``dataset/default.yaml``.

The Backbone
------------

In most cases, the backbone represents the transformer model you'd like to use when fine-tuning or pre-training on downstream tasks.
In the case of iGPT, this is a GPT model, but can be swapped out with other implementations hence the importance to define this as a separate entity.

The backbone can be anything required, in our case just a ``nn.Module`` defining a simple GPT structure:

.. code-block:: python

    class GPT2(nn.Module):

    def __init__(self, embed_dim, num_heads, num_layers, num_positions, num_vocab, num_classes):
        super(GPT2, self).__init__()

        self.embed_dim = embed_dim

        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        self.token_embeddings = nn.Embedding(num_vocab, embed_dim)
        self.position_embeddings = nn.Embedding(num_positions, embed_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Block(embed_dim, num_heads))

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vocab, bias=False)
        self.clf_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, classify=False):
        ...

The full file can be seen at ``lightning_transformers/task/vision/gpt2.py``.

To instantiate the object,
