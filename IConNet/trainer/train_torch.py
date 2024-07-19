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
from typing import Iterable
from .model_wrapper import ModelWrapper
from .data_torch import SimpleDataModule as DataModule
from torch.utils.data import random_split, DataLoader
from ..utils.config import DatasetConfig
from ..dataset import DEFAULTS
from ..acov.model import SCB8 as SCB
from ..acov.audio_vqvae import VqVaeLoss, VqVaeClsLoss

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
            loss_ratio: VqVaeClsLoss,
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

        self.loss_ratio = loss_ratio


    def setup(self, model: SCB, lr=1e-3):
        self.lr = lr
        self.model = model
        
        self.train_losses = []
        self.train_losses_detail: list[VqVaeClsLoss] = []
        self.test_accuracy = []
        self.optimizer = optim.RAdam(self.model.parameters(), lr=lr)
    
        self.pbar = None
        self.current_epoch = 0
        self.current_step = 0
        self.best_epoch = 0
        self.best_accuracy = 0.
        self.best_model_path = None
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
                self.model.parameters(), self.gradient_clip_val)
            
    def compute_loss(self, losses_detail: list[VqVaeClsLoss]) -> tuple[Tensor, VqVaeClsLoss]:
        values = []
        loss = torch.tensor(0., requires_grad=True)
        for key in VqVaeClsLoss._fields:
            v = torch.stack([getattr(ld, key) for ld in losses_detail]).mean()
            loss = loss + v * getattr(self.loss_ratio, key)
            values.append(v.item())
        loss_detail = VqVaeClsLoss(*values)
        return loss, loss_detail

    def train_step(self):
        device = self.device
        self.model.train()
        losses_detail: list[VqVaeClsLoss] = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            _mem_before = torch.cuda.memory_allocated()
            data = data.to(device)
            target = target.to(device)
            logits, vqvae_loss = self.model.train_embedding_cls(data)
            del data
            gc.collect()
            torch.cuda.empty_cache()
            loss_cls = F.cross_entropy(logits.squeeze(), target)
            loss_detail = VqVaeClsLoss(*vqvae_loss, loss_cls=loss_cls)
            losses_detail.append(loss_detail)
            _mem_during = torch.cuda.memory_allocated()
            del target
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
                    self.eval_step(loss=loss, message=message)
            self._update_progress_bar()
                                    
    
    def gen_log_message(self, batch_idx, loss, loss_detail, memory=None):
        # if memory is not None:
            # message = f'Mem before-during-after: {memory}\n'
        message = ""
        progress = f"{batch_idx * self.batch_size}/{self.train_data_size} "
        progress += f"({100. * batch_idx / self.train_batches:.0f}%)"
        message += f"Epoch: {self.current_epoch}\tLoss: {loss:.3f}"
        values = [f"{k}={loss_detail[i]:.3f}" for i,k in enumerate(loss_detail._fields)]
        message += f" [{', '.join(values)}]\t"
        return message

    def log_eval(self, loss, acc, message, result_dict):
        print(message)
        suffix = f"epoch={self.current_epoch}.step={self.current_step}."
        suffix += f"loss={loss:.3f}.val_acc={acc:.3f}.pt"
        result_path = f"{self.log_dir}val_result.{suffix}"

        torch.save(result_dict, result_path)
        if acc > self.best_accuracy:
            self.best_epoch = self.current_epoch
            self.best_accuracy = acc 
            model_path = f"{self.log_dir}model.{suffix}"
            torch.save(self.model.state_dict(), model_path)
            self.best_model_path = model_path
            print(f"Saved new best model: {self.best_model_path}")
            
    # TODO: log csv (skip?)

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
            
            # metrics.update(probs, target)
            # metrics_detail.update(probs, target)
            confusion_matrix.update(probs, target)
            
            del target
            gc.collect()
            torch.cuda.empty_cache()

            self._update_progress_bar()
        print(f'Correct: {correct}/{total} ({correct/total:.4f})')
        return metrics, metrics_detail, confusion_matrix

    def fit(self, n_epoch=10):
        device = self.device
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

                metrics, metrics_details, confusion_matrix = self.test_step()
                # pprint(metrics.compute())
                # pprint(metrics_details.compute())
                pprint(confusion_matrix.compute())

                self.scheduler.step()
        self.model.to('cpu')       

    def save(self, checkpoint_path):
        torch.save(self.model.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))

    def load_best_model(self):
        if self.best_model_path is not None:
            self.model.load_state_dict(torch.load(self.best_model_path))
        else:
            print('best_model_path not found')

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







