from pathlib import Path

import pytest

from tests import CACHE_PATH  # GitHub Actions use this path to cache datasets.


@pytest.fixture(scope="session")
def hf_cache_path() -> Path:
    datadir = Path(CACHE_PATH)
    return datadir / "huggingface"
