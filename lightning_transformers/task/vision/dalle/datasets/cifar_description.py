from typing import Dict
from warnings import warn

from clip import clip, tokenize
from clip.simple_tokenizer import SimpleTokenizer
from PIL import Image
from pl_bolts.datamodules import CIFAR10DataModule

try:
    from torchvision import transforms as transform_lib
except ImportError:
    warn(
        "You want to use `torchvision` which is not installed yet,"  # pragma: no-cover
        " install it with `pip install torchvision`."
    )
    _TORCHVISION_AVAILABLE = False
else:
    _TORCHVISION_AVAILABLE = True


class CIFARDescriptionDataModule(CIFAR10DataModule):
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
            clip._tokenizer = self.tokenizer
            text = tokenize(text, context_length)
            return text[0]

        self.EXTRA_ARGS = {"target_transform": target_transform}

    @property
    def model_data_args(self) -> Dict:
        return {"num_text_tokens": len(self.tokenizer.encoder)}

    def default_transforms(self):
        return transform_lib.Compose([
            transform_lib.Resize(256, interpolation=Image.BICUBIC),
            transform_lib.ToTensor(),
        ])
