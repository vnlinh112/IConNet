import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from IConNet.IConNet.utils.config import Config, TrainSklearnConfig, TrainPyTorchConfig

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="train", name="sklearn", node=TrainSklearnConfig)
cs.store(group="train", name="pytorch", node=TrainPyTorchConfig)

@hydra.main(version_base=None, config_path="config", config_name="default")
def my_app(cfg : Config) -> None:
    print(OmegaConf.to_yaml(cfg))
    # print(cfg["dataset"])
    # print(cfg.dataset.classnames[0])
    # print(cfg.dataset.tasks["4_emotions"])
    # print(cfg.model)
    

if __name__ == "__main__":
    my_app()