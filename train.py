import hydra
from omegaconf import DictConfig, OmegaConf
from IConNet.utils.config import Config
from IConNet.trainer import train, train_cv
from coolname import generate_slug


@hydra.main(version_base=None, config_path="config", config_name="default")
def run(config : Config) -> None:
    print(OmegaConf.to_yaml(config))

    if not config.exp.slug:
        slug = generate_slug(2)
    else:
        slug = config.exp.slug 
        
    if config.train.cross_validation:
        train_cv(
            config, 
            experiment_prefix=config.exp.experiment_prefix,
            experiment_suffix=slug)
    else:
        train(
            config, 
            experiment_prefix=config.exp.experiment_prefix,
            experiment_suffix=slug)
    
if __name__ == "__main__":
    run()