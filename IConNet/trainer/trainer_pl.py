import lightning as L
from .model_pl import ModelPLClassification as LightningModel
from .model_pl import PredictionWriter
import torchmetrics
from lightning.pytorch.loggers import (
    TensorBoardLogger, WandbLogger, CSVLogger
    )
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import (
    GradientAccumulationScheduler,
    ModelCheckpoint
    )
from .dataloader import DataModule, DataModuleKFold
from ..utils.config import Config, get_valid_path
import wandb

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
torch.set_float32_matmul_precision("high")

def get_loggers(
        dataset, 
        feature, 
        model_name, 
        wandb_project="",
        experiment_prefix="", 
        experiment_suffix="", 
        log_dir: str = '_logs/',
        tensorboard = False
):  
    
    experiment_dir = f"{dataset}" 
    if experiment_prefix is not None and len(experiment_prefix) > 0:
        experiment_dir = f"{experiment_prefix}.{experiment_dir}"
    if wandb_project is None or len(wandb_project) == 0:
        wandb_project = experiment_dir
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
    if tensorboard:
        tb_logger = TensorBoardLogger(
            save_dir=f"{experiment_dir}", name=run_name)
        loggers = [tb_logger, csv_logger, wandb_logger]
    else:
        loggers = [csv_logger, wandb_logger]
    return loggers, run_dir

def train(
        config: Config,
        data_dir: str,
        data: DataModule = None,
        experiment_prefix="", # dryrun, test, hpc, ...
        experiment_suffix="", # red-racoon
        log_dir: str = '_logs/',
        loggers = None,
        run_dir = None
        ):
    
    L.seed_everything(config.train.random_seed, workers=True)
    pin_memory = config.train.accelerator == 'gpu'
    
    if loggers is None:
        loggers, run_dir = get_loggers(
            dataset = config.dataset.name,
            feature = config.dataset.feature_name,
            model_name = config.model.name,
            log_dir = get_valid_path(log_dir),
            experiment_prefix = experiment_prefix,
            experiment_suffix = experiment_suffix
        )

    if data is None:
        data = DataModule(
            config=config.dataset,
            data_dir=data_dir,
            num_workers=config.train.num_workers,
            batch_size=config.train.batch_size,
            pin_memory=pin_memory)
        data.prepare_data()
    data.setup()

    litmodel = LightningModel(
        config.model, 
        n_input=data.num_channels, 
        n_output=data.num_classes,
        train_config=config.train,
        classnames=data.classnames
        )
    
    callbacks = []
    if config.train.accumulate_grad:
        if config.train.accumulate_grad_scheduler:
            callbacks += [GradientAccumulationScheduler(
                scheduling=config.train.accumulate_grad_scheduler)]
        else:
            callbacks += [GradientAccumulationScheduler(
                scheduling={0:8})]
        
    callbacks += [ModelCheckpoint(
        monitor='val_UF1',
        mode='max',
        save_top_k=config.train.checkpoint_save_top_k, # -1: save all
        save_last=True,
        filename='{epoch}-{val_UA:.2f}-{val_UF1:.2f}-{val_WF1:.2f}',
        auto_insert_metric_name=True,
        # save_weights_only=True,
        every_n_epochs=1
    )]        
    callbacks += [PredictionWriter(
        output_dir=run_dir, write_interval="epoch")]

    if config.train.early_stopping:
        callbacks += [EarlyStopping(
            monitor="val_acc", 
            min_delta=0.00, 
            patience=10, 
            verbose=False, 
            mode="max")]

    trainer = L.Trainer(
        max_epochs=config.train.max_epochs,
        min_epochs=config.train.min_epochs,
        callbacks=callbacks,
        accelerator=config.train.accelerator,
        devices=config.train.devices,
        num_nodes=config.train.num_nodes,
        gradient_clip_val=1.,
        val_check_interval=config.train.val_check_interval,  # 0.5: twice per epoch
        precision=config.train.precision,           
        logger=loggers,
        deterministic=True,
        detect_anomaly=config.train.detect_anomaly,
        # inference_mode=False # use torch.no_grad
    )

    trainer.fit(
        litmodel, 
        train_dataloaders = data.train_dataloader(), 
        val_dataloaders = data.val_dataloader(),
        ckpt_path="last"
        )
    
    data.setup("test")
    trainer.test(
        dataloaders=data.test_dataloader(),
        ckpt_path="best")
    
    data.setup("predict")
    trainer.predict(
        dataloaders=data.predict_dataloader(),
        ckpt_path="best",
        return_predictions=False
    )

    wandb.finish()
    
    
def train_cv(
        config: Config, 
        data_dir: str,
        experiment_prefix="",
        experiment_suffix="",
        log_dir: str = '_logs/'):
    num_folds = config.train.num_folds
    pin_memory = config.train.accelerator == 'gpu'
    
    wandb_project =  config.dataset.name + ".cv"
    if experiment_prefix is not None and len(experiment_prefix) > 0:
        wandb_project = f'{experiment_prefix}.{wandb_project}'

    log_dir = get_valid_path(log_dir) + wandb_project
    prefix = f'{config.model.name}.{experiment_suffix}'
    for i in range(num_folds):
        fold_number = i
        suffix = f'fold{fold_number+1}.{experiment_suffix}'

        loggers, run_dir = get_loggers(
            dataset = config.dataset.name,
            feature = config.dataset.feature_name,
            model_name = config.model.name,
            log_dir = log_dir,
            experiment_prefix = prefix,
            experiment_suffix = suffix,
            wandb_project="",
        )

        data = DataModuleKFold(
            config=config.dataset,
            data_dir=data_dir,
            fold_number=fold_number, 
            num_splits=num_folds, 
            split_seed=config.train.random_seed,
            batch_size=config.train.batch_size,
            num_workers=config.train.num_workers,
            pin_memory=pin_memory)
        data.prepare_data()

        train(
            data=data,
            config=config,
            experiment_prefix=prefix,
            experiment_suffix=suffix,
            log_dir=log_dir,
            loggers=loggers,
            run_dir=run_dir
        )

def test(litmodel, x, y):
    pred_model = litmodel.model
    pred_model.eval()
    preds = pred_model(x)
    acc = torchmetrics.functional.accuracy(preds, y)
    print(acc)
