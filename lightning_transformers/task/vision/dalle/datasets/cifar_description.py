from lightning_transformers.task.vision.dalle.clip.clip import tokenize
from lightning_transformers.task.vision.dalle.clip.simple_tokenizer import SimpleTokenizer
from lightning_transformers.task.vision.dalle.datasets.cifar_generate import CIFARGenerateDataModule


class CIFARDescriptionDataModule(CIFARGenerateDataModule):
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    """
    Adds a description to each image based on the label to use as a toy dataset for text/image pairs
    """

    def __init__(self, *args, tokenizer: SimpleTokenizer, context_length: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.context_length = context_length

        def target_transform(target):
            prefix = "Generate an image of a"
            label = CIFARDescriptionDataModule.labels[target]
            if label.startswith("a"):
                prefix += "n"  # todo: may be overkill for grammar sake...
            text = f"{prefix} {label}"
            return tokenize(text, tokenizer, context_length)

        self.extra_args = {"target_transform": target_transform}
