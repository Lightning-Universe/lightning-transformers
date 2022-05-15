"""Root package info."""
import os

__ROOT_DIR__ = os.path.dirname(os.path.dirname(__file__))

__version__ = "0.1.0"
__author__ = "PyTorchLightning et al."
__author_email__ = "name@pytorchlightning.ai"
__license__ = "Apache-2.0"
__copyright__ = f"Copyright (c) 2020-2020, {__author__}."
__homepage__ = "https://github.com/PyTorchLightning/lightning-transformers"
__docs__ = "PyTorch Lightning Transformers."
__long_doc__ = """
Lightning Transformers provides `LightningModules`, `LightningDataModules` and `Strategies` to use
HuggingFace Transformers with the PyTorch Lightning Trainer.
"""

from lightning_transformers.core.nlp import HFTransformerDataConfig  # noqa: F401
from lightning_transformers.core.nlp.seq2seq import Seq2SeqDataConfig  # noqa: F401
from lightning_transformers.task.nlp.language_modeling import (  # noqa: F401
    LanguageModelingDataModule,
    LanguageModelingTransformer,
)
from lightning_transformers.task.nlp.language_modeling.config import LanguageModelingDataConfig  # noqa: F401
from lightning_transformers.task.nlp.masked_language_modeling import (  # noqa: F401
    MaskedLanguageModelingDataModule,
    MaskedLanguageModelingTransformer,
)
from lightning_transformers.task.nlp.masked_language_modeling.config import (  # noqa: F401
    MaskedLanguageModelingDataConfig,
)
from lightning_transformers.task.nlp.multiple_choice import (  # noqa: F401
    MultipleChoiceDataModule,
    MultipleChoiceTransformer,
    RaceMultipleChoiceDataModule,
    SwagMultipleChoiceDataModule,
)
from lightning_transformers.task.nlp.question_answering import (  # noqa: F401
    QuestionAnsweringDataModule,
    QuestionAnsweringTransformer,
)
from lightning_transformers.task.nlp.question_answering.config import QuestionAnsweringDataConfig  # noqa: F401
from lightning_transformers.task.nlp.question_answering.datasets import SquadDataModule  # noqa: F401
from lightning_transformers.task.nlp.summarization import (  # noqa: F401
    CNNDailyMailSummarizationDataModule,
    SummarizationDataModule,
    SummarizationTransformer,
    XsumSummarizationDataModule,
)
from lightning_transformers.task.nlp.summarization.config import SummarizationConfig  # noqa: F401
from lightning_transformers.task.nlp.text_classification import (  # noqa: F401
    TextClassificationDataModule,
    TextClassificationTransformer,
)
from lightning_transformers.task.nlp.token_classification import (  # noqa: F401
    TokenClassificationDataModule,
    TokenClassificationTransformer,
)
from lightning_transformers.task.nlp.token_classification.config import TokenClassificationDataConfig  # noqa: F401
from lightning_transformers.task.nlp.translation import (  # noqa: F401
    TranslationDataModule,
    TranslationTransformer,
    WMT16TranslationDataModule,
)
from lightning_transformers.task.nlp.translation.config import TranslationConfig, TranslationDataConfig  # noqa: F401
