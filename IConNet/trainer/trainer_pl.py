import pytorch_lightning as pl
from .model_pl import ModelPLClassification as LightningModel
import torchmetrics
from pytorch_lightning.loggers import (
    TensorBoardLogger, WandbLogger, CSVLogger
)
from lightning.callbacks.early_stopping import EarlyStopping
from .dataloader import DataModule, DataModuleKFold
pl.seed_everything(42, workers=True)

from ..utils.config import Config, get_valid_path

def train(
        config: Config,
        data: DataModule = None,
        experiment_prefix="", # test
        experiment_suffix="", # fold1
        log_dir: str = '_logs/'):
    
    if data is None:
        data = DataModule(
            config=config.data,
            data_dir=config.data_dir)
        data.prepare_data()
        data.setup()

    dataset = data.config.name
    feature= data.config.feature_name
    model_name = config.model.name

    log_dir = get_valid_path(config.log_dir)

    exp = f"{model_name}.{dataset}.{feature}"               # M13.ravdess.audio16k
    exp = f"{experiment_prefix}.{exp}.{experiment_suffix}"  # SER4.M13.ravdess.audio16k.fold1
    wandb_logger = WandbLogger(
        project="test-ser-23", 
        save_dir=f"{log_dir}wandb_logs", name=exp)
    csv_logger = CSVLogger(
        save_dir=f"{log_dir}lightning_logs", name=exp)
    tb_logger = TensorBoardLogger(
        save_dir=f"{log_dir}lightning_logs", name=exp)
    loggers = [tb_logger, csv_logger, wandb_logger]

    litmodel = LightningModel(config.model)

    if config.early_stopping:
        early_stop_callback = [EarlyStopping(
            monitor="val_acc", 
            min_delta=0.00, 
            patience=3, 
            verbose=False, 
            mode="max")]
    else:
        early_stop_callback = None

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        min_epochs=config.min_epochs,
        callbacks=early_stop_callback,
        gpus=config.gpus,
        gradient_clip_val=1.,
        val_check_interval=config.val_check_interval,  # 0.5: twice per epoch
        precision=config.precision,            # floating precision 16 makes ~5x faster
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
        dataloaders=data.test_dataloader(),
        ckpt_path="best")
    
    
def train_cv(config: Config, experiment_prefix=""):
    num_folds = config.num_folds
    for i in range(num_folds):
        fold_number = i+1
        data = DataModuleKFold(
            config=config.data,
            data_dir=config.data_dir,
            fold_number=fold_number, 
            num_splits=num_folds, 
            split_seed=config.random_seed)
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
