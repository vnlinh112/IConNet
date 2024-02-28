import pytorch_lightning as pl
from .model_pl import ModelPLClassification
import torchmetrics
from pytorch_lightning.loggers import (
    TensorBoardLogger, WandbLogger, CSVLogger
)
from lightning.callbacks.early_stopping import EarlyStopping
from .dataloader import DataModule, DataModuleKFold
pl.seed_everything(42, workers=True)

def train(config, data: DataModule):

    dataset = "ravdess"
    num_class = 4
    use_label = "label_emotion"
    use_all_data = False 
    feature = 'feature.mfcc128'
    model_name = "M13"

    exp = f"{dataset}{num_class}_{feature}_{model_name}"
    wandb_logger = WandbLogger(
        project="test-ser-23", save_dir="wandb_logs", name=exp)
    csv_logger = CSVLogger(
        save_dir="lightning_logs", name=exp)
    tb_logger = TensorBoardLogger(
        save_dir="lightning_logs", name=exp)
    loggers = [tb_logger, csv_logger, wandb_logger]

    litmodel = ModelPLClassification(config)

    early_stop_callback = EarlyStopping(
        monitor="val_acc", 
        min_delta=0.00, 
        patience=3, 
        verbose=False, 
        mode="max")

    trainer = pl.Trainer(
        max_epochs=100,
        min_epochs=10,
        callbacks=[early_stop_callback],
        gpus=1,
        gradient_clip_val=1.,
        val_check_interval=0.5,  # 0.5: twice per epoch
        precision=16,            # floating precision 16 makes ~5x faster
        logger=loggers,
        deterministic=True,
    )
    trainer.fit(
        litmodel, 
        train_dataloaders = data.train_dataloader(), 
        val_dataloaders = data.val_dataloader(),
        ckpt_path="last")
    
    trainer.test(
        dataloaders=data.test_dataloader(),
        ckpt_path="best")
    
def train_cv(config):
    num_folds = 5
    split_seed = 42
    for i in range(num_folds):
        datamodule = DataModuleKFold(
            fold_number=i+1, 
            num_splits=num_folds, 
            split_seed=split_seed)
        datamodule.prepare_data()
        datamodule.setup()
        train(config, datamodule)

def test(litmodel, x, y):
    pred_model = litmodel.model
    pred_model.eval()
    preds = pred_model(x)
    acc = torchmetrics.functional.accuracy(preds, y)
    print(acc)
