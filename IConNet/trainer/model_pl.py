import torch
import lightning as L
from .metrics import get_metrics
from .model_wrapper import ModelWrapper

class ModelPLClassification(L.LightningModule):
    def __init__(self, config, n_input, n_output):
        super().__init__()
        self.config = config
        self.n_input = n_input
        self.n_output = n_output
        self.model = ModelWrapper(self.config.name).init_model(
            self.config, n_input=n_input, n_output=n_output)
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_metrics = get_metrics(n_output, 'train')
        self.val_metrics = get_metrics(n_output, 'val')
        self.test_metrics = get_metrics(n_output, 'test')

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
        output = self.val_metrics(logits, y)
        self.log_dict(output)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)    
        self.log('test_loss', loss)
        output = self.test_metrics(logits, y)
        self.log_dict(output)
        return loss

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