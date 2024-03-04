import torch
import lightning as L
from .metrics import get_metrics, get_detail_metrics
from .model_wrapper import ModelWrapper
from lightning.pytorch.callbacks import BasePredictionWriter

import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import os

class ModelPLClassification(L.LightningModule):
    def __init__(self, config, n_input, n_output, 
                 classnames=None):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.n_input = n_input
        self.n_output = n_output
        self.classnames = classnames
        self.model = ModelWrapper(self.config.name).init_model(
            self.config, n_input=n_input, n_output=n_output)
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_metrics = get_metrics(n_output, 'train')
        self.val_metrics = get_metrics(n_output, 'val')
        self.test_metrics = get_metrics(n_output, 'test')
        # test_metrics_detail, test_confusion_matrix = get_detail_metrics(n_output, 'test')
        # self.test_metrics_detail = test_metrics_detail
        # self.test_confusion_matrix = test_confusion_matrix

    def forward(self, x):
        logits = self.model(x)
        return logits  

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)      
         # Logging to TensorBoard by default
        self.log('train_loss', loss)
        output = self.train_metrics(logits, y)
        self.log_dict(output)        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch  
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss)
        self.val_metrics.update(logits, y)
        return loss
    
    def on_validation_epoch_end(self):
        output = self.val_metrics.compute()
        self.log_dict(output, on_step=False, on_epoch=True)
        self.val_metrics.reset()  
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        output = self.test_metrics(logits, y)
        self.log_dict(output)
        # self.test_metrics_detail(logits, y)
        # self.test_confusion_matrix(logits, y)
        return logits
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
    
    # def on_test_epoch_end(self):
    #     test_confusion_matrix = self.test_confusion_matrix.compute()
    #     class_names = self.classnames
    #     df_cm = pd.DataFrame(
    #         test_confusion_matrix.cpu().numpy() , 
    #         index = class_names, 
    #         columns = class_names)   
    #     norm =  np.sum(df_cm, axis=1)
    #     normalized_cm = (df_cm.T/norm).T
    #     f, ax = plt.subplots(figsize = (15,10)) 
    #     sns.heatmap(normalized_cm, annot=True, ax=ax)
    #     wandb.log({"plot": wandb.Image(f) })
    #     self.test_confusion_matrix.reset()  

    def configure_optimizers(self):
        """
        To overcome nan loss with fp16 (Ref: PyTorchLightning/issues/2673)
        Prepare optimizer and schedule (linear warmup and decay)
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.RAdam(optimizer_grouped_parameters, lr=1e-4)
        return optimizer
    

class PredictionWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(
            self, trainer, pl_module, 
            prediction, batch_indices, 
            batch, batch_idx, dataloader_idx
            ):
        out_path = os.path.join(
            self.output_dir, 
            dataloader_idx, 
            f"{batch_idx}.pt")
        torch.save(prediction, out_path)

    def write_on_epoch_end(
            self, trainer, pl_module, 
            predictions, batch_indices
            ):
        out_path = os.path.join(self.output_dir, "predictions.pt")
        torch.save(predictions, out_path)

    
class PredictionWriterDDP(BasePredictionWriter):

    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        out_path = os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt")
        torch.save(predictions, out_path)