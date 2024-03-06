from dataclasses import dataclass, field
from typing import Iterable, Optional, Dict, Union
from enum import auto
from strenum import StrEnum

def get_optional_config_value(config_value: None):
    if config_value is None or config_value=='None' or config_value==False:
        return None
    return config_value

def get_valid_path(path: str):
    if path.endswith('/'):
        return path 
    return path + '/'

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

class AcceleratorType(StrEnum):
    cpu = auto()
    gpu = auto()
    tpu = auto()
    ipu = auto()
    hpu = auto()
    auto = auto()

class TrainerStrategyType(StrEnum):
    ddp = auto()
    fsdp = auto()
    ddp_spawn = auto()
    deepspeed = auto()
    auto = auto()

@dataclass
class DatasetConfig:
    name: str = "crema_d"
    root: str = "crema_d/"
    audio_dir: str = "full_release/"
    feature_dir: str = "preprocessed/"
    label_name: str = "label_emotion"
    feature_name: str = "audio16k"
    num_classes: int = 6
    label_values: Iterable[str] = field(
        default_factory = lambda: [
            "neu", "hap", "sad", "ang", "fea", "dis"])
    classnames: Iterable[str] = field(
        default_factory = lambda: [
            "neu", "hap", "sad", "ang", "fea", "dis"])
    target_labels: Optional[Iterable[str]] = field(
        default_factory = lambda: [
            "ang", "neu", "sad", "hap"])
    
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
class TrainPyTorchConfig(TrainConfig):
    name: str = "torch demo"
    batch_size: int = 16
    n_epoch: int = 2
    early_stopping: bool = False
    accumulate_grad: bool = False
    accumulate_grad_scheduler: Dict[int,int] = field(
        default_factory= lambda: {
            0: 8, 50: 4, 80: 1
        }
    )
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
    max_epochs: int=100
    min_epochs: int=10
    detect_anomaly: bool = False
    # strategy: Optional[TrainerStrategyType] = None
    accelerator: AcceleratorType=AcceleratorType.cpu
    devices: int=1
    num_nodes: int=1 # num gpu nodes
    num_workers: int=8 
    val_check_interval: float=0.5
    precision = 32
    cross_validation: bool = False
    num_folds: int=5
    random_seed: int=42 

@dataclass
class Config:
    data_dir: str
    log_dir: str = "_logs/"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainPyTorchConfig)