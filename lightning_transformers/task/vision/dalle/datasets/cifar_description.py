from lightning_transformers.task.vision.dalle.datasets.cifar import CIFARTransformerDataModule


class CIFARDescriptionDataModule(CIFARTransformerDataModule):
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    """
    Adds a description to each image based on the label to use as a toy dataset for text/image pairs
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_args = {"target_transform": self.target_transform}

    @staticmethod
    def target_transform(target) -> str:
        prefix = "Generate an image of a"
        label = CIFARDescriptionDataModule.labels[target]
        if label.startswith("a"):
            prefix += "n"  # todo: may be overkill for grammar sake...
        return f"{prefix} {label}"
