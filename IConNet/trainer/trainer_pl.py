import lightning as L
from .model_pl import ModelPLClassification as LightningModel
import torchmetrics
from lightning.pytorch.loggers import (
    TensorBoardLogger, WandbLogger, CSVLogger
)
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from .dataloader import DataModule, DataModuleKFold
from ..utils.config import Config, get_valid_path

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

def get_loggers(
        dataset, 
        feature, 
        model_name, 
        experiment_prefix="", 
        experiment_suffix="", 
        log_dir: str = '_logs/'
):  
    
    experiment_dir = f"{dataset}" 
    if experiment_prefix is not None and len(experiment_prefix) > 0:
        experiment_dir = f"{experiment_prefix}.{experiment_dir}"
    wandb_project = "test-ser-23" #experiment_dir
    experiment_dir = f'{log_dir}{experiment_dir}/'

    run_name = f"{model_name}" 
    if experiment_suffix is not None and len(experiment_suffix) > 0:
        run_name = f"{run_name}.{experiment_suffix}"

    run_dir = f'{experiment_dir}{run_name}/' 

    if not os.path.isdir(run_dir):
        os.makedirs(run_dir, exist_ok=True)
        print(f'Created log dir: {run_dir}')
    else:
        print(f'Writing to existing log dir: {run_dir}')

    print(f'Logging to wandb project: {wandb_project}')

    wandb_logger = WandbLogger(
        project=wandb_project,
        save_dir=f"{run_dir}", name=run_name) 
    csv_logger = CSVLogger(
        save_dir=f"{experiment_dir}", name=run_name)
    tb_logger = TensorBoardLogger(
        save_dir=f"{experiment_dir}", name=run_name)
    loggers = [tb_logger, csv_logger, wandb_logger]
    return loggers

def train(
        config: Config,
        data: DataModule = None,
        experiment_prefix="", # dryrun, test, hpc, ...
        experiment_suffix="", # red-racoon
        log_dir: str = '_logs/'
        ):
    
    L.seed_everything(config.train.random_seed, workers=True)
    pin_memory = config.train.accelerator == 'gpu'
    
    if data is None:
        data = DataModule(
            config=config.dataset,
            data_dir=config.data_dir,
            pin_memory=pin_memory)
        data.prepare_data()
        data.setup()
    
    loggers = get_loggers(
        dataset = data.config.name,
        feature = data.config.feature_name,
        model_name = config.model.name,
        log_dir = get_valid_path(log_dir),
        experiment_prefix = experiment_prefix,
        experiment_suffix = experiment_suffix
    )

    litmodel = LightningModel(
        config.model, 
        n_input=data.num_channels, 
        n_output=data.num_classes)

    if config.train.early_stopping:
        early_stop_callback = [EarlyStopping(
            monitor="val_acc", 
            min_delta=0.00, 
            patience=3, 
            verbose=False, 
            mode="max")]
    else:
        early_stop_callback = None

    trainer = L.Trainer(
        max_epochs=config.train.max_epochs,
        min_epochs=config.train.min_epochs,
        callbacks=early_stop_callback,
        accelerator=config.train.accelerator,
        devices=config.train.devices,
        gradient_clip_val=1.,
        val_check_interval=config.train.val_check_interval,  # 0.5: twice per epoch
        precision=config.train.precision,            # floating precision 16-mixed makes ~5x faster
        logger=loggers,
        deterministic=True,
    )

    trainer.fit(
        litmodel, 
        train_dataloaders = data.train_dataloader(), 
        val_dataloaders = data.val_dataloader(),
        # ckpt_path="last"
        )
    
    trainer.test(
        dataloaders= data.val_dataloader(), #data.test_dataloader(),
        ckpt_path="best")
    
    
def train_cv(config: Config, experiment_prefix=""):
    num_folds = config.train.num_folds
    pin_memory = config.train.accelerator == 'gpu'
    for i in range(num_folds):
        fold_number = i+1
        data = DataModuleKFold(
            config=config.dataset,
            data_dir=config.data_dir,
            fold_number=fold_number, 
            num_splits=num_folds, 
            split_seed=config.train.random_seed,
            pin_memory=pin_memory)
        data.prepare_data()
        data.setup()

        train(
            data=data,
            config=config,
            experiment_prefix=experiment_prefix,
            experiment_suffix=f'fold{fold_number}'
        )

def test(litmodel, x, y):
    pred_model = litmodel.model
    pred_model.eval()
    preds = pred_model(x)
    acc = torchmetrics.functional.accuracy(preds, y)
    print(acc)
