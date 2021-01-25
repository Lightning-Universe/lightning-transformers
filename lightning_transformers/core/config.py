from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class BaseConfig:
    def __new__(cls, *args, **kwargs):
        # Adapted from: https://stackoverflow.com/a/63291704
        try:
            initializer = cls.__initializer
        except AttributeError:
            cls.__initializer = initializer = cls.__init__
            cls.__init__ = lambda *a, **k: None

        if not hasattr(cls, "__annotations__"):
            extras = kwargs
            kwargs = {}
        else:
            extras = {kwargs.pop(name) for name in kwargs if name not in cls.__annotations__}

        new = object.__new__(cls)
        initializer(new, *args, **kwargs)
        new._repr_extras = []
        for k, v in extras.items():
            setattr(new, k, v)
            new._repr_extras.append(k)
        return new

    def __repr__(self) -> str:
        attrs = [f"{k}={getattr(self, k)!r}" for k in self._repr_extras]
        return f"{self.__class__.__name__}({', '.join(attrs)})"

    def asdict(self) -> Dict:
        def blocklist(s: str) -> bool:
            s = str(s)
            return s.startswith("_") and s not in ("_target_", "_recursive_")

        return {k: v for k, v in self.__dict__.items() if not blocklist(k)}

    def __eq__(self, other) -> bool:
        if not isinstance(other, BaseConfig):
            return False
        return self.asdict() == other.asdict()


@dataclass
class HydraConfig:
    _target_: Optional[str] = None
    _target_config_: Optional[str] = None


@dataclass
class TransformerDataConfig(HydraConfig):
    batch_size: int = 32
    num_workers: int = 0


@dataclass
class OptimizerConfig(HydraConfig):
    lr: float = 1e-3
    weight_decay: float = 0.0


@dataclass
class SchedulerConfig(HydraConfig):
    num_training_steps: int = -1
    num_warmup_steps: float = 0.1


@dataclass
class TrainerConfig(HydraConfig):
    ...


@dataclass
class TaskConfig(HydraConfig):
    ...
