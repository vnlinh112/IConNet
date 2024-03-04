import hydra
from omegaconf import DictConfig, OmegaConf
from IConNet.utils.config import Config
from IConNet.trainer import train, train_cv
from coolname import generate_slug
from datetime import datetime


@hydra.main(version_base=None, config_path="config", config_name="default")
def run(config : Config) -> None:
    print(OmegaConf.to_yaml(config))
    slug = generate_slug(2)
    date = datetime.now().strftime("%m%d%Y")
    prefix = f'cv.{slug}.{date}'
    train_cv(
        config, 
        experiment_prefix=prefix)
    
if __name__ == "__main__":
    run()