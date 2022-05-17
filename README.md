<div align="center">

<img src="docs/source/_static/images/logo.png" width="500px">

**Flexible components pairing :hugs: Transformers with [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)**

______________________________________________________________________

<p align="center">
  <a href="https://lightning-transformers.readthedocs.io/">Docs</a> •
  <a href="#community">Community</a>
</p>

______________________________________________________________________

</div>

## Installation

```bash
pip install lightning-transformers
```

<details>
<summary>From Source</summary>

```bash
git clone https://github.com/PyTorchLightning/lightning-transformers.git
cd lightning-transformers
pip install .
```

</details>

______________________________________________________________________

## What is Lightning-Transformers

Lightning Transformers provides `LightningModules`, `LightningDataModules` and `Strategies` to use :hugs: Transformers with the [PyTorch Lightning Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html).

## Quick Recipes

#### Train [bert-base-cased](https://huggingface.co/bert-base-cased) on the [CARER](https://huggingface.co/datasets/emotion) emotion dataset using the Text Classification task.

```python
import pytorch_lightning as pl
from transformers import AutoTokenizer

from lightning_transformers.task.nlp.text_classification import (
    TextClassificationDataModule,
    TextClassificationTransformer,
    TextClassificationDataConfig,
)

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="bert-base-cased"
)
dm = TextClassificationDataModule(
    cfg=TextClassificationDataConfig(
        batch_size=1,
        dataset_name="emotion",
        max_length=512,
    ),
    tokenizer=tokenizer,
)
model = TextClassificationTransformer(pretrained_model_name_or_path="bert-base-cased")

trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=1)

trainer.fit(model, dm)
```

#### Train a pre-trained [mt5-base](https://huggingface.co/google/mt5-base) backbone on the [WMT16](https://huggingface.co/datasets/wmt16) dataset using the Translation task.

```python
import pytorch_lightning as pl
from transformers import AutoTokenizer

from lightning_transformers.task.nlp.translation import (
    TranslationTransformer,
    WMT16TranslationDataModule,
    TranslationConfig,
    TranslationDataConfig,
)

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="google/mt5-base"
)
model = TranslationTransformer(
    pretrained_model_name_or_path="google/mt5-base",
    cfg=TranslationConfig(
        n_gram=4,
        smooth=False,
        val_target_max_length=142,
        num_beams=None,
        compute_generate_metrics=True,
    ),
)
dm = WMT16TranslationDataModule(
    cfg=TranslationDataConfig(
        dataset_name="wmt16",
        # WMT translation datasets: ['cs-en', 'de-en', 'fi-en', 'ro-en', 'ru-en', 'tr-en']
        dataset_config_name="ro-en",
        source_language="en",
        target_language="ro",
        max_source_length=128,
        max_target_length=128,
    ),
    tokenizer=tokenizer,
)
trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=1)

trainer.fit(model, dm)
```

Lightning Transformers supports a bunch of :hugs: tasks and datasets. See the [documentation](https://lightning-transformers.readthedocs.io/en/latest/).

## Contribute

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Community

For help or questions, join our huge community on [Slack](https://www.pytorchlightning.ai/community)!
