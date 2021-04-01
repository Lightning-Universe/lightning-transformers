.. _new-task:

Customizing Tasks
=================

You can build your own Lightning Transformers task for any custom use case. To create your own task you will need to:

* Implement a Dataset
* Implement a Backbone
* (Optional) Implement a Tokenizer
* Implement a Task

In most cases you will need to implement a subset of them. There is also interconnection between
the pieces; for example when your model requires knowing how many classes are within your dataset. We cover all these details below.

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

Implementing a Dataset
----------------------

The dataset defines the data transforms, plus the data you'd like to train, validate and test on. In our example we build the CIFAR10 dataset for iGPT.

The base class for Lightning Transformer datasets is the  ``TransformerDataModule`` class, which is a thin layer on top of the LightningDataModule class, exposing all the Lightning Data Hooks as standard. See `LightningDataModule <https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html>`_ for more details.
Additional, the base class provides a few helper functions, such as ``model_data_kwargs`` which allows us to define any data specific arguments we'd like the task to be aware of at initialize time.

In some cases you do not need to define a module, and could use a pre-set LightningDataModule out the box.

Here is a simplified example of the iGPT DataModule.

.. code-block:: python

    class ImageGPTDataModule(TransformerDataModule):

        def __init__(self, cfg: ImageGPTDataConfig = ImageGPTDataConfig()):
            super().__init__(cfg=cfg)

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
        def model_data_kwargs(self) -> Dict:
            """
            Override to provide the model with additional kwargs.
            This is useful to provide the number of classes/pixels to the model or any other data specific args
            Returns: Dict of args
            """
            return {}

In our case, most of our logic fits within the ``prepare_data`` and ``setup`` function. Here is a simplistic view of the logic.

.. code-block:: python

    ...
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        # Saves files to directory
        self.dataset_cls(self.cfg.data_dir, train=True, download=True)
        self.dataset_cls(self.cfg.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        ...
        if stage == "fit" or stage is None:
            # Setup image dataset using data transforms
            train_transforms = self.train_transforms
            val_transforms = self.test_transforms

            dataset_train = self.dataset_cls(self.cfg.data_dir, train=True, transform=train_transforms)
            dataset_val = self.dataset_cls(self.cfg.data_dir, train=True, transform=val_transforms)

            ...
            # Split dataset into train/val
            self.dataset_train, _ = random_split(dataset_train, lengths, generator=generator.manual_seed(0))
            _, self.dataset_val = random_split(dataset_val, lengths, generator=generator.manual_seed(0))
        # Optionally process the test dataset
        if stage == "test" or stage is None:
            self.dataset_test = self.dataset_cls(self.cfg.data_dir, train=False, transform=self.test_transforms)

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
      - /dataset/default
    _target_: lightning_transformers.task.vision.igpt.data.ImageGPTDataModule
    cfg:
      dataset: cifar10
      data_dir: ./


Here we define the ``_target_`` class we'll like to instantiate (our DataModule that we just implemented) as well as any specific arguments.
We also inherit from a default config which can be found in ``dataset/default.yaml``.

.. _implement-backbone:

Implementing a Backbone
-----------------------

In most cases, the backbone represents the transformer model you'd like to use when fine-tuning or pre-training on downstream tasks.
In the case of iGPT, this is a GPT model, but can be swapped out with other implementations hence the importance to define this as a separate entity.

The backbone can be anything. In our case, the backbone is a ``nn.Module`` defining a simple GPT structure:

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

To instantiate the object, we have to define a config. This config can be seen in ``conf/backbone/vision/igpt/gpt.yaml``.

.. code-block:: yaml

    # @package backbone
    _target_: lightning_transformers.task.vision.igpt.GPT2
    embed_dim: 16
    num_heads: 2
    num_layers: 8
    num_vocab: 16
    num_classes: 10

We define the ``_target_`` class we'd like to instantiate, and parameters for our backbone model. For setting the backbone, we can do this via the cmdline or via hydra defaults. See :ref:`implement-task`.

.. _implement-tokenizer:

(Optional) Implement a Tokenizer
--------------------------------

For many NLP models, a tokenizer will need to be defined. In many cases you can use pre-built tokenizers which saves having to train your own model.

For example here is the config for HF tokenizers found in ``conf/tokenizer/autotokenizer.yaml``.

.. code-block:: yaml

    # @package tokenizer
    _target_: transformers.AutoTokenizer.from_pretrained
    pretrained_model_name_or_path: ${backbone.pretrained_model_name_or_path}
    use_fast: true

Here we instantiate the ``_target_`` function, and pass in the necessary arguments. One of them is a shared parameter with the backbone; the model name (i.e bert-base-cased).

In the case of iGPT, no tokenizer this is purely pixel based training.

.. _implement-task:

Implement a Task
----------------

The task contains the logic required to train, validation and test the model. The base class for this in most cases is ``TaskTransformer``, which contains a few helper functions on top of the standard ``pl.LightningModule`` class.

Below is a simplified version of the file found at ``lightning_transformers/task/vision/igpt/model.py``

.. code-block:: python

    class GenerativePixelsTransformer(TaskTransformer):

        def __init__(
            self,
            num_pixels: int, # Number of input pixels
            backbone: Any, # Our backbone config
            optimizer: OptimizerConfig,
            scheduler: SchedulerConfig,
            instantiator: Optional[Instantiator] = None, # Hydra instantiator object to instantiate configs
            classify: bool = False,
        ):
            # Instantiate the backbone from the config
            backbone = instantiator.instantiate(backbone, num_positions=num_pixels * num_pixels)
            super().__init__(backbone, optimizer, scheduler, instantiator)
            self.save_hyperparameters()
            self.classify = classify

            self.criterion = nn.CrossEntropyLoss()

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch

            x = quantize(x, self.centroids)
            x = _to_sequence(x)

            if self.classify:
                clf_logits, logits = self.model(x, classify=True)
                clf_loss = self.criterion(clf_logits, y)
                gen_loss = self.criterion(logits.view(-1, logits.size(-1)), x.view(-1))
                # joint loss for classification
                loss = clf_loss + gen_loss
            else:
                logits = self.model(x)
                loss = self.criterion(logits.view(-1, logits.size(-1)), x.view(-1))

            logs = {"loss": loss}
            return {"loss": loss, "log": logs}

        def validation_step(self, batch, batch_idx):
            ...

        def test_step(self, batch, batch_idx):
            return self.validation_step(batch, batch_idx)

The ``TaskTransformer`` class requires as input an optimizer config, scheduler config, and the backbone.

In the above case, the backbone is a config object as defined in :ref:`implement-backbone`, and we use the ``instantiator`` which is a helper object for us to instantiate the object via the config.
This allows us to remain agnostic to the backbone. Specifically, if we wanted to implement another backbone such as BERT, the task would not need change.

Finally we define the config. This can be found in ``conf/task/vision/igpt.yaml``

.. code-block:: yaml

    # @package task
    defaults:
      - /task/default # Use the defaults from the default task config
      - /backbone@_group_: vision/igpt/gpt2 # default to vision gpt2
    _target_: lightning_transformers.task.vision.igpt.GenerativePixelsTransformer
    num_pixels: 32

Within the default task config (found at ``conf/task/default/yaml``) we define the optimizer/scheduler/backbone config via interpolation, which are then passed to our ``_target_`` class when instantiated in our task.

We also add a default backbone to our defaults list, which means when the task is specified at runtime, we default to the GPT2 backbone.

Running the Task
----------------

With all the pieces, we're able to use the CLI or ``train.py`` script to run our task.

.. code-block:: bash

    python train.py +task=vision/igpt +dataset=vision/igpt/cifar trainer.gpus=1

Note we do not need to define the backbone, as we've made our backbone default within the Task config.
