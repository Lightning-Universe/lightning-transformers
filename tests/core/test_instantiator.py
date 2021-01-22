import pytest
from omegaconf import DictConfig

from lightning_transformers.core.config import TransformerDataConfig
from lightning_transformers.core.instantiator import HydraInstantiator, Instantiator


def test_instantiator_raises():
    instantiator = Instantiator()
    with pytest.raises(NotImplementedError):
        instantiator.test  # noqa: pointless-statement


@pytest.mark.parametrize(
    ["x", "expected"],
    [
        # basic cases
        ({"foo": 1}, {"foo": 1}),
        ({"bar": {"foo": 1}}, {"bar": {"foo": 1}}),
        # check interpolation is resolved
        ({"bar": {"foo": 1}, "baz": "${bar.foo}"}, {"bar": {"foo": 1}, "baz": 1}),
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


@pytest.mark.parametrize(
    "x",
    [
        {
            "batch_size": 1,
            "_target_": "test",
            "_target_config_": "lightning_transformers.core.config.TransformerDataConfig",
            "non_existent_key": True,
        },
    ],
)
def test_hydrainstantiator_dictconfig_to_dataclass_raises(x):
    instantiator = HydraInstantiator()
    cfg = DictConfig(x)
    with pytest.raises(
        KeyError, match="unexpected key 'non_existent_key' in the config file for target config TransformerDataConfig"
    ):
        instantiator.dictconfig_to_dataclass(cfg)
