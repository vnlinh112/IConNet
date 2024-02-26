import hydra
from omegaconf import DictConfig, OmegaConf, MISSING
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

from typing import Iterable, List, Optional, Dict, Union
from enum import auto
from strenum import StrEnum

def get_optional_config_value(config_value: None):
    if config_value is None or config_value=='None' or config_value==False:
        return None
    return config_value

class SklearnSolver(StrEnum):
    lbfgs = auto()
    sgd = auto()
    adam = auto()

class SklearnLR(StrEnum):
    constant = auto()
    invscaling = auto()
    adaptive = auto()

class PyTorchOptimizer(StrEnum):
    SGD = auto()
    Adam = auto()
    AdamW = auto()
    RAdam= auto()

class PyTorchLR(StrEnum):
    StepLR = auto()
    OneCyleLR = auto()
    ExponentialLR = auto()

class ResidualConnectionType(StrEnum):
    stack = auto()
    concat = auto()
    add = auto()
    contract = auto()
    gated_add = auto()
    gated_contract = auto()

class PoolingType(StrEnum):
    max = auto()
    min = auto()
    mean = auto()
    sum = auto()

@dataclass
class DatasetConfig:
    name: str = "meld"
    root: str = "../data/meld/"
    audio: str = "../data/meld/audio16k"
    preprocessed: str = "../data/meld/features/"
    classnames: Iterable[str] = field(
        default_factory = lambda: [
            "neutral", "happy", "sad", "angry"])
    tasks = None
    
@dataclass
class FeBlockConfig:
    n_block: int = 2
    n_channel: Iterable[int] = (128, 128)
    kernel_size: Iterable[int] = (511, 127)
    stride: Iterable[int] = (2, 4)
    window_k: Iterable[int] = (2, 3)
    residual_connection_type: ResidualConnectionType = ResidualConnectionType.gated_contract
    pooling: PoolingType = PoolingType.max

@dataclass
class ClsBlockConfig:
    """automatically have 1 additional output layer"""
    n_block: int = 1
    n_hidden_dim: Iterable[int] = (320)

@dataclass
class ModelConfig:
    name: str = "M10"
    description: str = "FirConv"
    fe: FeBlockConfig = field(default_factory=FeBlockConfig)
    cls: ClsBlockConfig = field(default_factory=ClsBlockConfig)


@dataclass
class TrainConfig:
    name: str = "demo"
    batch_size: int = 16
    learning_rate_init: float = 0.001
    n_epoch: int = 2

@dataclass
class TrainSklearnConfig(TrainConfig):
    name: str = "demo"
    batch_size: int = 16
    solver: SklearnSolver = SklearnSolver.adam
    learning_rate: SklearnLR = SklearnLR.adaptive
    learning_rate_init: float = 0.001
    l2_reg: float = 0.001
    max_iter: int = 10
    early_stopping: bool = True

@dataclass
class TrainPyTorchConfig(TrainConfig):
    name: str = "demo"
    batch_size: int = 16
    n_epoch: int = 2
    early_stopping: bool = False
    log_interval: int = 40
    optimizer: PyTorchOptimizer = PyTorchOptimizer.RAdam
    optimizer_kwargs: Dict[str, Union[str,int,float,bool]] = field(
        default_factory = lambda: {
            "weight_decay": 0.0001
        })
    learning_rate_init: float = 0.001
    lr_scheduler: Optional[PyTorchLR] = PyTorchLR.OneCyleLR
    lr_scheduler_kwargs: Dict[str, Union[str,int,float,bool]] = field(
        default_factory = lambda: {
            "step_size": 40
        })

@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainPyTorchConfig)