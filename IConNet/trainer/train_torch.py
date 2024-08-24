import os
import argparse
import gc
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.optim as optim
import sys
from tqdm import tqdm
import traceback
# torch.autograd.set_detect_anomaly(True)
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassPrecision, 
    MulticlassRecall, MulticlassF1Score, 
    MulticlassConfusionMatrix,
    MulticlassAUROC
)
from torchmetrics import MetricCollection
from pprint import pprint
from typing import Iterable, Union, Optional, Literal
from .model_wrapper import ModelWrapper
from .data_torch import SimpleDataModule as DataModule
from torch.utils.data import random_split, DataLoader
from ..utils.config import DatasetConfig
from ..dataset import DEFAULTS
from ..acov.model import SCB
from ..acov.audio_vqvae import VqVaeLoss, VqVaeClsLoss
from ..acov.audio_vqmix import AudioVQMixClsLoss
from ..acov.audio_vqmix import AudioVQMixClsLoss

class Trainer:
    def __init__(
            self, 
            # config: DatasetConfig, 
            batch_size, log_dir, experiment_prefix, device,
            eval_ratio: float=0.2,
            gradient_clip_val: float=0,
            accumulate_grad_batches: int=1,

    ):
        """
        eval_ratio: extract val set from test_set to speed up training. Default: 0.2
        gradient_clip_val: set > 0 to apply gradient clipping. Default: 0
        accumulate_grad_batches: set > 1 to accumulate gradient: Default: 1
        """
        # self.config = config
        self.log_dir = log_dir
        self.batch_size = batch_size # config.train.batch_size
        self.experiment_prefix = experiment_prefix
        self.device = device
        self.labels = DEFAULTS["labels"] # config.labels
        self.num_classes = len(self.labels)

        self.eval_ratio = eval_ratio
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches 
        self.val_check_interval = 0.5

    def prepare(
            self, 
            train_loader: DataLoader, 
            test_loader: DataLoader,
            loss_ratio: Optional[Union[VqVaeClsLoss, AudioVQMixClsLoss]]=None,
            eval_loader=None,
            batch_size=None,
        ):
        if batch_size is not None:
            self.batch_size = batch_size
        self.train_loader = train_loader
        self.test_loader = test_loader 
        self.eval_loader = eval_loader
        self.train_data_size = len(train_loader.dataset)
        self.test_data_size = len(test_loader.dataset)
        self.train_batches = len(train_loader)
        self.test_batches = len(test_loader)

        self.val_check_batches = int(self.val_check_interval * self.train_batches)
        self.total_batches = self.train_batches + self.test_batches
        self.steps_per_epoch = self.train_batches // self.accumulate_grad_batches
        self.loss_ratio = loss_ratio

    @property
    def learnable_parameters(self):
        params = [p for p in self.model.parameters() if p.requires_grad==True]
        return params

    def setup(self, model: SCB, lr=1e-3):
        self.lr = lr
        self.model = model
        
        self.train_losses = []
        self.train_losses_detail: list[Union[VqVaeClsLoss, AudioVQMixClsLoss]] = []
        self.test_accuracy = []
        self.val_accuracy = []
    
        self.pbar = None
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_epoch = 0
        self.best_val_accuracy = 0.
        self.best_val_model_path = None
        self.best_test_epoch = 0
        self.best_test_accuracy = 0
        self.best_test_model_path = None
        self.pbar_update = 1/self.total_batches

        n = self.num_classes
        self.metrics = MetricCollection({
            'acc_unweighted': MulticlassAccuracy(num_classes=n, average='macro'), 
            'acc_weighted': MulticlassAccuracy(num_classes=n, average='weighted'),            
            'f1s_unweighted': MulticlassF1Score(num_classes=n, average='macro'),  
            'f1s_weighted': MulticlassF1Score(num_classes=n, average='weighted'),  
            'uar': MulticlassRecall(num_classes=n, average='macro'), 
            'wap': MulticlassPrecision(num_classes=n, average='weighted'),
            'rocauc': MulticlassAUROC(num_classes=n, average='macro', thresholds=None),    
        })
        self.metrics_detail = MetricCollection({  
            'f1s_detail': MulticlassF1Score(num_classes=n, average=None),  
            'acc_detail': MulticlassAccuracy(num_classes=n, average=None), 
            'precision_detail': MulticlassPrecision(num_classes=n, average=None),
            'recall_detail': MulticlassRecall(num_classes=n, average=None), 
            'rocauc_detail': MulticlassAUROC(num_classes=n, average=None, thresholds=None),
        })
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=n)

    def _update_progress_bar(self):
        if self.pbar is not None:
            self.pbar.update(self.pbar_update)

    def clip_gradient(self):
        if self.gradient_clip_val > 0:
            nn.utils.clip_grad.clip_grad_norm_(
                self.learnable_parameters, self.gradient_clip_val)
            
    def compute_loss(
            self, losses_detail: list[Union[VqVaeClsLoss, AudioVQMixClsLoss]]
        ) -> tuple[Tensor, Union[VqVaeClsLoss, AudioVQMixClsLoss]]:
        values = []
        loss = torch.tensor(0., requires_grad=True)
        loss_type = type(losses_detail[0])
        for key in loss_type._fields:
            v = torch.stack([getattr(ld, key) for ld in losses_detail]).mean()
            loss = loss + v * getattr(self.loss_ratio, key)
            values.append(v.item())
        loss_detail = loss_type(*values)
        loss_detail = loss_type(*values)
        return loss, loss_detail

    def train_step(
            self, 
            self_supervised=False, 
            train_task: Literal[
                'embedding', 'projector', 
                'embedding_projector']='embedding'
        ):
        device = self.device
        self.model.train()
        losses_detail: list[Union[VqVaeClsLoss, AudioVQMixClsLoss]] = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            _mem_before = torch.cuda.memory_allocated()
            data = data.to(device)
            if self_supervised and train_task=='embedding':
                loss_detail = self.model.train_embedding(data)
            elif self_supervised and train_task=='projector':
                loss_detail = self.model.train_projector(data)
            elif self_supervised and train_task=='embedding_projector':
                loss_detail = self.model.train_embedding_projector(data)
            elif train_task=='projector':
                target = target.to(device)
                _, loss_detail = self.model.train_projector_cls(data, target)
            elif train_task=='embedding':
                target = target.to(device)
                _, loss_detail = self.model.train_embedding_cls(data, target)
            else: # embedding_projector
                target = target.to(device)
                _, loss_detail = self.model.train_embedding_projector_cls(data, target)
            del data
            gc.collect()
            torch.cuda.empty_cache()
            losses_detail.append(loss_detail)
            _mem_during = torch.cuda.memory_allocated()
            gc.collect()
            torch.cuda.empty_cache()

            if batch_idx % self.accumulate_grad_batches:
                self.optimizer.zero_grad()
                loss, loss_detail = self.compute_loss(losses_detail)
                loss.backward()
                self.clip_gradient()
                self.optimizer.step()
                _mem_after = torch.cuda.memory_allocated()
                self.current_step += 1

                loss = loss.item()
                self.train_losses.append(loss)
                self.train_losses_detail.append(loss_detail)

                losses_detail = []
                if batch_idx % self.val_check_batches == 0: 
                    message = self.gen_log_message(
                        batch_idx, loss, loss_detail,
                        memory=(_mem_before, _mem_during, _mem_after)
                    )    
                    if self_supervised: 
                        self.log_eval(
                            loss=loss, acc=0., 
                            message=message)
                    else:
                        self.eval_step(loss=loss, message=message)
            self._update_progress_bar()
                                    
    
    def gen_log_message(self, batch_idx, loss, loss_detail=None, memory=None):
        # if memory is not None:
            # message = f'Mem before-during-after: {memory}\n'
        message = ""
        progress = f"{batch_idx * self.batch_size}/{self.train_data_size} "
        progress += f"({100. * batch_idx / self.train_batches:.0f}%)"
        message += f"Epoch: {self.current_epoch}\tLoss: {loss:.3f}"
        if loss_detail is not None:
            values = [f"{k}={loss_detail[i]:.3f}" for i,k in enumerate(loss_detail._fields)]
            message += f" [{', '.join(values)}]\t"
        else: 
            message += "\t"
        return message

    def log_eval(self, loss, acc, message, result_dict=None):
        self.val_accuracy.append(acc)
        print(message)
        suffix = f"epoch={self.current_epoch}.step={self.current_step}."
        suffix += f"loss={loss:.3f}.val_acc={acc:.3f}.pt"
        if result_dict is not None:
            result_path = f"{self.log_dir}val_result.{suffix}"
            torch.save(result_dict, result_path)
        if acc > self.best_val_accuracy:
            self.best_val_epoch = self.current_epoch
            self.best_val_accuracy = acc 
            model_path = f"{self.log_dir}model.{suffix}"
            torch.save(self.model.state_dict(), model_path)
            self.best_val_model_path = model_path
            print(f"Saved new best val model: {self.best_val_model_path}")

    @torch.no_grad
    def eval_step(self, loss, message):
        device = self.device
        self.model.eval()
        self.model = self.model.to(device)
        correct = 0
        total = 0
        logits_batches = []
        latents_batches = []
        ypreds_batches = []
        ytrues_batches = []
        is_correct_batches = []

        if self.eval_loader is not None:
            data_loader = self.eval_loader
            n_iter = len(data_loader)
        else:
            data_loader = self.test_loader
            n_iter = int(len(data_loader) * self.eval_ratio)
        data_generator = iter(data_loader)
        for i in range(n_iter):
            data, target = next(data_generator)
            total += len(target)
            data = data.to(device)
            target = target.to(device)
            logits, latents = self.model.classify(data)
            latents_batches += [latents.detach().cpu()]
            logits_batches += [logits.detach().cpu()]
            del data, latents

            gc.collect()
            torch.cuda.empty_cache()
            probs = F.softmax(logits.squeeze(), dim=-1)
            preds = probs.argmax(dim=-1)
            is_correct = preds.eq(target).detach().cpu()
            correct += sum(is_correct).item()
            ypreds_batches += [preds.detach().cpu()]
            ytrues_batches += [target.detach().cpu()]
            is_correct_batches += [is_correct]
            del target, is_correct, preds, probs
            gc.collect()
            torch.cuda.empty_cache()

        codebook = self.model.embedding_filters.detach().cpu()
        logits = torch.concat(logits_batches, dim=0)
        latents = torch.concat(latents_batches, dim=0)
        ypreds = torch.concat(ypreds_batches, dim=0)
        ytrues = torch.concat(ytrues_batches, dim=0)
        is_correct = torch.concat(is_correct_batches, dim=0)
        acc = correct / total
        result_dict = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'loss': loss,
            'val_acc': acc,
            'val_correct': correct,
            'val_total': total,
            'codebook': codebook,
            'logits': logits,
            'latents': latents,
            'ypreds': ypreds,
            'ytrues': ytrues,
            'is_correct': is_correct,
        }
        message += f"Val_acc: {correct}/{total} ({100.*acc:.2f}%)\n"
        self.log_eval(
            loss=loss, acc=acc, 
            message=message, 
            result_dict=result_dict)
        self.model.train()
    
    @torch.no_grad
    def test_step(self):
        device = self.device
        metrics = self.metrics.clone().to(device)
        metrics_detail = self.metrics_detail.clone().to(device)
        confusion_matrix = self.confusion_matrix.clone().to(device)
        self.model = self.model.to(device)
        self.model.eval()
        correct = 0
        total = 0
        for data, target in self.test_loader:
            total += len(target)
          
            data = data.to(device)
            target = target.to(device)
            preds, probs = self.model.predict(data)
            del data
            gc.collect()
            torch.cuda.empty_cache()
            correct += preds.eq(target).sum().item()
            
            metrics.update(probs, target)
            metrics_detail.update(probs, target)
            confusion_matrix.update(probs, target)
            
            del target
            gc.collect()
            torch.cuda.empty_cache()

            self._update_progress_bar()
        acc = correct/total
        print(f'Correct: {correct}/{total} ({acc:.4f})')
        self.test_accuracy.append(acc)
        if acc > self.best_test_accuracy:
            self.best_test_epoch = self.current_epoch
            self.best_test_accuracy = acc 
            suffix = f"epoch={self.current_epoch}.step={self.current_step}"
            suffix += f".test_acc={acc:.4f}.pt"
            model_path = f"{self.log_dir}model.{suffix}"
            torch.save(self.model.state_dict(), model_path)
            self.best_test_model_path = model_path
            print(f"Saved new best test model: {self.best_test_model_path}")

        self.model.train()        
        return metrics, metrics_detail, confusion_matrix

    def fit(
            self, n_epoch=10, 
            self_supervised=False,
            train_task: Literal[
                'embedding', 'projector', 
                'embedding_projector']='embedding',
            contrastive_learning=False,
            loss_ratio=None,
            lr=None,
            test_n_epoch: Optional[int]=1,
            optimizer=None, scheduler=None,
            optimizer_with_regularizer=False):
        
        device = self.device
        has_test_step = not self_supervised
        if test_n_epoch is None:
            num_test_batches = 0
        else:
            num_test_batches = (self.test_batches//test_n_epoch)*has_test_step
        self.total_batches = self.train_batches + num_test_batches
        self.pbar_update = 1/self.total_batches
        if lr is not None:
            self.lr = lr
        if loss_ratio is not None:
            self.loss_ratio = loss_ratio
        if optimizer_with_regularizer:
            self.set_optimizer_with_regularizer(optim_cls=torch.optim.SGD)
        else:
            if optimizer is None:
                self.optimizer = optim.RAdam(self.learnable_parameters, lr=self.lr)
            else:
                self.optimizer = optimizer
            if scheduler is None:
                self.scheduler = optim.lr_scheduler.OneCycleLR(
                    self.optimizer, max_lr=0.1,
                    steps_per_epoch=self.steps_per_epoch, 
                    epochs=n_epoch)
            else:
                self.scheduler = scheduler
        self.model.to(device)
        with tqdm(total=n_epoch) as pbar:
            self.pbar = pbar
            for epoch in range(n_epoch):
                self.current_epoch += 1
                self.train_step(
                    self_supervised=self_supervised, 
                    train_task=train_task)
                
                is_test_epoch = (test_n_epoch is not None and test_n_epoch > 0 and epoch % test_n_epoch == 0)
                if is_test_epoch and self_supervised == True:
                    self.train_step(
                        self_supervised=False, 
                        train_task=train_task)
                
                self.scheduler.step()

                if is_test_epoch:
                    metrics, metrics_details, confusion_matrix = self.test_step()
                    pprint(metrics.cpu().compute())
                    pprint(metrics_details.cpu().compute())
                    pprint(confusion_matrix.cpu().compute())
                    self.save()        

        self.model.to('cpu')       


    def save(self, checkpoint_path=None):
        if not checkpoint_path:
            suffix = f"epoch={self.current_epoch}.step={self.current_step}.pt"
            checkpoint_path = f"{self.log_dir}model.{suffix}"
        torch.save(self.model.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))

    def load_best_model(self, val_model=False):
        if val_model==False and self.best_test_model_path is not None:
            self.model.load_state_dict(torch.load(self.best_test_model_path))
            print(f'Loaded: {self.best_test_model_path}')
        elif self.best_val_model_path is not None:
            self.model.load_state_dict(torch.load(self.best_val_model_path))
            print(f'Loaded: {self.best_val_model_path}')
        else:
            print('No best model found!')

    def set_optimizer_with_regularizer(self, optim_cls=torch.optim.SGD):    
        lr = self.lr
        weight_decay = max(lr/10, 1e-6)
        max_lr = min(lr*100, 0.1)
        steps_per_epoch = self.steps_per_epoch
        step_size_up = min(steps_per_epoch, 2000)
        decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ] 
        self.optimizer = optim_cls(
            optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay)       
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer, 
            base_lr=lr, 
            max_lr=max_lr, 
            step_size_up=step_size_up,
            mode="triangular2")

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

class Trainer_SCB10(Trainer):
    def __init__(
            self, 
            # config: DatasetConfig, 
            batch_size, log_dir, experiment_prefix, device,
            eval_ratio: float=0.2,
            gradient_clip_val: float=0,
            accumulate_grad_batches: int=1,

    ):
        super().__init__(
            batch_size, log_dir, experiment_prefix, 
            device, eval_ratio, gradient_clip_val, accumulate_grad_batches
        )

    @torch.no_grad
    def eval_step(self, loss, message):
        device = self.device
        self.model.eval()
        self.model = self.model.to(device)
        correct = 0
        total = 0

        if self.eval_loader is not None:
            data_loader = self.eval_loader
            n_iter = len(data_loader)
        else:
            data_loader = self.test_loader
            n_iter = int(len(data_loader) * self.eval_ratio)
        data_generator = iter(data_loader)
        for i in range(n_iter):
            data, target = next(data_generator)
            total += len(target)
            data = data.to(device)
            target = target.to(device)
            preds, _ = self.model.predict(data)
            is_correct = preds.eq(target).detach().cpu()
            correct += sum(is_correct).item()
            del data, target
            gc.collect()
            torch.cuda.empty_cache()

        acc = correct / total
        message += f"Val_acc: {correct}/{total} ({100.*acc:.2f}%)\n"
        self.log_eval(
            loss=loss, acc=acc, 
            message=message)
        self.model.train()

class Trainer_custom_model(Trainer_SCB10):
    def __init__(
            self, 
            batch_size, log_dir, experiment_prefix, device,
            eval_ratio: float=0.2,
            gradient_clip_val: float=0,
            accumulate_grad_batches: int=1,
    ):
        super().__init__(
            batch_size, log_dir, experiment_prefix, device,
            eval_ratio, gradient_clip_val, accumulate_grad_batches)

    def setup(self, model, lr=1e-3):
        super().setup(model=None, lr=lr)
        self.model = model
        self.train_losses_detail = []

    def train_step(self):
        device = self.device
        self.model.train()
        loss = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(device)
            target = target.to(device)
            logits = self.model(data)
            loss = loss + F.cross_entropy(logits.squeeze(), target)
            del data, target
            gc.collect()
            torch.cuda.empty_cache()
            if batch_idx % self.accumulate_grad_batches:
                self.optimizer.zero_grad()
                loss.backward()
                self.clip_gradient()
                self.optimizer.step()
                self.current_step += 1
                loss = loss.item()
                self.train_losses.append(loss)
                if batch_idx % self.val_check_batches == 0: 
                    message = self.gen_log_message(
                        batch_idx, loss, loss_detail=None,
                        memory=None
                    )    
                    self.eval_step(loss=loss, message=message)
                loss = torch.tensor(0., dtype=torch.float64, requires_grad=True)
            self._update_progress_bar()
            
    def fit(self, n_epoch=10, lr=None, test_n_epoch=1):
        device = self.device
        self.total_batches = self.train_batches + self.test_batches // test_n_epoch
        self.pbar_update = 1/self.total_batches
        if lr is not None:
            self.lr = lr
        self.optimizer = optim.RAdam(self.learnable_parameters, lr=self.lr)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=0.1,
            steps_per_epoch=self.train_batches, 
            epochs=n_epoch)
        self.model.to(device)
        with tqdm(total=n_epoch) as pbar:
            self.pbar = pbar
            for epoch in range(n_epoch):
                self.current_epoch += 1
                self.train_step()
                self.scheduler.step()
                is_test_epoch = (test_n_epoch is not None and test_n_epoch > 0 and epoch % test_n_epoch == 0)   
                if is_test_epoch:
                    metrics, metrics_details, confusion_matrix = self.test_step()
                    pprint(metrics.compute())
                    pprint(metrics_details.compute())
                    pprint(confusion_matrix.compute())           
        self.model.to('cpu')  


def get_dataloader(config: DatasetConfig, data_dir, batch_size=None):
    data = DataModule(config, data_dir, batch_size=batch_size)
    data.prepare_data() 
    data.setup("fit")
    train_loader = data.train_dataloader()
    test_loader = data.val_dataloader()
    batch_size = data.batch_size
    return train_loader, test_loader, batch_size


def main(
        dataset_config: DatasetConfig, 
        data_dir, log_dir, experiment_prefix, 
        batch_size=None, n_epoch=2):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    trainer = Trainer(log_dir, experiment_prefix, device)
    train_loader, test_loader, batch_size = get_dataloader(dataset_config, data_dir)
    trainer.prepare(train_loader, test_loader, batch_size)
    trainer.setup()
    trainer.fit(n_epoch=n_epoch)
    
    # metrics, metrics_details, confusion_matrix = trainer.test_step()
    # pprint(metrics.compute())
    # pprint(metrics_details.compute())
    # pprint(confusion_matrix.compute())

    # trainer.save(checkpoint_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str,
        default='./experiments/crema_d.4.librosa_nnaudio.pca/config.yml',
        help='Config file')
    parser.add_argument(
        '--data_dir', type=str,
        default='../data/data_preprocessed/',
        help='Data directory that contains datasets with preprossed data')
    parser.add_argument(
        '--log_dir', type=str,
        default='_logs/',
        help='Model checkpoints directory')
    args = parser.parse_args()

    main(args.config, args.data_dir, args.checkpoint_path)







