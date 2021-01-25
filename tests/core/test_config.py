from lightning_transformers.core.config import BaseConfig


def test_baseconfig():
    x = BaseConfig(foo="bar", other=BaseConfig(test=True))
    x.baz = 3
    # check __dict__
    assert x.asdict() == {"foo": "bar", "baz": 3, "other": BaseConfig(test=True)}
    # check __repr__
    assert repr(x) == "BaseConfig(foo='bar', other=BaseConfig(test=True))"
    # check __eq__
    assert BaseConfig(x=1) == BaseConfig(x=1)
    assert BaseConfig(x=1) != BaseConfig(x=2)
    assert BaseConfig(x=1) != BaseConfig(y=1)
