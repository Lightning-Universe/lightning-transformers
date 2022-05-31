from torch.utils.data import DataLoader, _DatasetKind
from torch.utils.data.dataloader import _InfiniteConstantSampler


class IterableDataLoader(DataLoader):
    """Class to properly wrap the `datasets.IterableDataset` class.

    Unfortunately this class does not inherit from the `torch.data.IterableDataset` class, so has to be handled
    specially.
    """

    def __init__(self, *args, **kwargs):
        kwargs["sampler"] = _InfiniteConstantSampler()
        super().__init__(*args, **kwargs)
        self._dataset_kind = _DatasetKind.Iterable
