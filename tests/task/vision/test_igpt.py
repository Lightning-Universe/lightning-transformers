import sys

import pytest
from pytorch_lightning.utilities import _module_available


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
@pytest.mark.skipif(not _module_available("torchvision"), reason="Requires torchvision to run")
def test_smoke_train_e2e(script_runner):
    script_runner.cache_dir = script_runner.datadir / "igpt" / "cifar"
    script_runner.cache_dir.mkdir(parents=True, exist_ok=True)
    script_runner.train([
        "+task='vision/igpt'",
        "+dataset='vision/igpt/cifar'",
        'backbone.num_layers=1',
        f'dataset.cfg.data_dir={script_runner.cache_dir}',
        'trainer.fast_dev_run=1',
    ])
