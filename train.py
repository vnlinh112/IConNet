import hydra
from omegaconf import DictConfig, OmegaConf
from IConNet.utils.config import Config
from IConNet.trainer import train, train_cv


@hydra.main(version_base=None, config_path="config", config_name="default")
def run(config : Config) -> None:
    print(OmegaConf.to_yaml(config))
    train(config)
    
if __name__ == "__main__":
    run()