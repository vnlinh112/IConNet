import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from IConNet.IConNet.utils.config import Config
from IConNet.trainer import train, train_cv

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

@hydra.main(version_base=None, config_path="config", config_name="default")
def run(config : Config) -> None:
    print(OmegaConf.to_yaml(config))
    train_cv(config)
    
if __name__ == "__main__":
    run()