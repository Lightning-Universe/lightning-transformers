from typing import Dict
from warnings import warn

from PIL import Image

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

from pl_bolts.datamodules import CIFAR10DataModule


class CIFARTransformerDataModule(CIFAR10DataModule):
    """
    Adds a description to each image based on the label to use as a toy dataset for text/image pairs
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def model_data_args(self) -> Dict:
        # todo we should be provided this class as inheritence...
        return {"image_size": 256}

    def default_transforms(self):
        return transform_lib.Compose(
            [
                transform_lib.ToTensor(),
                transform_lib.Resize(256, interpolation=Image.BICUBIC),
            ]
        )
