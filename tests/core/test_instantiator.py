import pytest
from omegaconf import DictConfig

from lightning_transformers.core.config import BaseConfig, TransformerDataConfig
from lightning_transformers.core.instantiator import HydraInstantiator


@pytest.mark.parametrize(
    ["x", "expected"],
    [
        # basic cases
        ({"foo": 1}, BaseConfig(foo=1)),
        ({"bar": {"foo": 1}}, BaseConfig(bar=BaseConfig(foo=1))),
        # check interpolation is resolved
        ({"bar": {"foo": 1}, "baz": "${bar.foo}"}, BaseConfig(bar=BaseConfig(foo=1), baz=1)),
        (
            {
                "batch_size": 1,
                "_target_": "test",
                "_target_config_": "lightning_transformers.core.config.TransformerDataConfig",
            },
            TransformerDataConfig(
                _target_="test",
                _target_config_="lightning_transformers.core.config.TransformerDataConfig",
                batch_size=1,
            ),
        ),
    ],
)
def test_hydrainstantiator_dictconfig_to_dataclass(x, expected):
    instantiator = HydraInstantiator()
    cfg = DictConfig(x)
    actual = instantiator.dictconfig_to_dataclass(cfg)
    assert actual == expected


def test_hydrainstantiator_dictconfig_to_dataclass_raises(x):
    instantiator = HydraInstantiator()
    cfg = DictConfig(
        {
            "batch_size": 1,
            "_target_": "test",
            "_target_config_": "lightning_transformers.core.config.TransformerDataConfig",
            "test_key": True,
        }
    )
    with pytest.raises(
        KeyError, match="unexpected key 'test_key' in the config file for target config TransformerDataConfig"
    ):
        instantiator.dictconfig_to_dataclass(cfg)
