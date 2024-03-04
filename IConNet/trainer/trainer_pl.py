import lightning as L
from .model_pl import ModelPLClassification as LightningModel
import torchmetrics
from lightning.pytorch.loggers import (
    TensorBoardLogger, WandbLogger, CSVLogger
)
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from .dataloader import DataModule, DataModuleKFold
L.seed_everything(42, workers=True)

from ..utils.config import Config, get_valid_path

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

def train(
        config: Config,
        data: DataModule = None,
        experiment_prefix="", # fold1
        experiment_suffix="", # red-racoon
        log_dir: str = '_logs/'):
    
    if data is None:
        data = DataModule(
            config=config.dataset,
            data_dir=config.data_dir,
            labels=config.labels)
        data.prepare_data()
        data.setup()

    dataset = data.config.name
    feature= data.config.feature_name
    model_name = config.model.name

    log_dir = get_valid_path(log_dir)

    # exp = f"{model_name}.{dataset}.{feature}"               # M13.ravdess.audio16k
    exp = f"{model_name}" 

    if experiment_prefix is not None and len(experiment_prefix) > 0:
        exp = f"{experiment_prefix}.{exp}"
    if experiment_suffix is not None and len(experiment_suffix) > 0:
        exp = f"{exp}.{experiment_suffix}"  # SER4.M13.ravdess.audio16k.fold1

    wandb_logger = WandbLogger(
        project="test-ser-23",  #TODO: change to prefix_dataset
        save_dir=f"{log_dir}", name=exp) 
    csv_logger = CSVLogger(
        save_dir=f"{log_dir}", name=dataset)
    tb_logger = TensorBoardLogger(
        save_dir=f"{log_dir}", name=dataset)
    loggers = [tb_logger, csv_logger, wandb_logger]

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
        precision=config.train.precision,            # floating precision 16 makes ~5x faster
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
    for i in range(num_folds):
        fold_number = i+1
        data = DataModuleKFold(
            config=config.dataset,
            data_dir=config.data_dir,
            labels=config.labels,
            fold_number=fold_number, 
            num_splits=num_folds, 
            split_seed=config.train.random_seed)
        data.prepare_data()
        data.setup()

        train(
            data=data,
            model_config=config.model,
            log_dir=config.log_dir,
            experiment_prefix=experiment_prefix,
            experiment_suffix=f'fold{fold_number}'
        )

def test(litmodel, x, y):
    pred_model = litmodel.model
    pred_model.eval()
    preds = pred_model(x)
    acc = torchmetrics.functional.accuracy(preds, y)
    print(acc)
