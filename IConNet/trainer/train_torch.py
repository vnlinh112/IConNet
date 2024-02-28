import os
import argparse
import gc
import torch
import torch.nn as nn
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
from .model_wrapper import ModelWrapper

class Trainer:
    def __init__(self, config, device):
        self.batch_size = config.train.batch_size
        self.device = device
        self.labels = config.labels
        self.num_classes = len(self.labels)

    def prepare(self, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader 
        self.train_loader_length = len(train_loader.dataset)
        self.test_loader_length = len(test_loader.dataset)

    def setup(self):
        self.model = ModelWrapper.pick_model(
            self.config.model.name)(self.config.model)
        self.train_losses = []
        self.test_accuracy = []
        self.optimizer = optim.RAdam(self.model.parameters(), lr=0.01)
        self.log_interval = 40
        self.pbar_update = 1 / (len(self.train_loader) + len(self.test_loader))

    def train_step(self, epoch, log_interval, 
            train_losses, pbar, pbar_update):
        device = self.device
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            try:
                _mem_before = torch.cuda.memory_allocated()
                data = data.to(device)
                target = target.to(device)
                output = self.model(data)
                del data
                gc.collect()
                torch.cuda.empty_cache()
                loss = F.cross_entropy(output.squeeze(), target)
                _mem_during = torch.cuda.memory_allocated()
                del target
                gc.collect()
                torch.cuda.empty_cache()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad.clip_grad_norm_(
                    self.model.parameters(), 1.0)
                self.optimizer.step()
                _mem_after = torch.cuda.memory_allocated()
                # print training stats
                if batch_idx % log_interval == 0:
                    print(f"Train Epoch: {epoch} [{batch_idx * self.batch_size}/{self.train_loader_length} ({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
                    print(f'Mem before-during-after: {_mem_before} {_mem_during} {_mem_after}')
                # update progress bar
                pbar.update(pbar_update)
                # record loss
                train_losses.append(loss.item())
            except Exception as e:
                print(f'data: {data.shape} => output: {output.shape} | target: {target.shape}')
                traceback.print_exc()

    @torch.no_grad
    def test_step(self, model, epoch):
        device = self.device
        model.eval()
        correct = 0
        total = 0
        for data, target in self.test_loader:
            total += len(target)
            data = data.to(device)
            target = target.to(device)
            output = model(data).squeeze()
            del data
            gc.collect()
            torch.cuda.empty_cache()
            probs = F.softmax(output, dim=-1)
            pred = probs.argmax(dim=-1)
            correct += pred.eq(target).sum().item()
            del target
            gc.collect()
            torch.cuda.empty_cache()
        acc = correct / self.test_loader_length
        print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{total} ({100. * acc:.0f}%)\n")
        return acc
    
    @torch.no_grad
    def evaluate(self, model, test_loader, pbar):
        device = self.device
        n = self.num_classes
        metrics = MetricCollection({
            'acc_unweighted': MulticlassAccuracy(num_classes=n, average='macro'), 
            'acc_weighted': MulticlassAccuracy(num_classes=n, average='weighted'), 
            
            'f1s_unweighted': MulticlassF1Score(num_classes=n, average='macro'),  
            'f1s_weighted': MulticlassF1Score(num_classes=n, average='weighted'),  

            'acc_weighted': MulticlassAccuracy(num_classes=n, average='weighted'), 
            'uar': MulticlassRecall(num_classes=n, average='macro'), 
            'wap': MulticlassPrecision(num_classes=n, average='weighted'),
            'rocauc': MulticlassAUROC(num_classes=n, average='macro', thresholds=None),    
            'f1s_detail': MulticlassF1Score(num_classes=n, average=None),  
            'acc_detail': MulticlassAccuracy(num_classes=n, average=None), 
            'precision_detail': MulticlassPrecision(num_classes=n, average=None),
            'recall_detail': MulticlassRecall(num_classes=n, average=None), 
            'rocauc_detail': MulticlassAUROC(num_classes=n, average=None, thresholds=None),
        }).to(device)
        confusion_matrix = MulticlassConfusionMatrix(num_classes=n).to(device)
        model.eval()
        correct = 0
        total = 0
        for data, target in test_loader:
            total += len(target)
            
            data = data.to(device)
            target = target.to(device)
            output = model(data).squeeze()
            del data
            gc.collect()
            torch.cuda.empty_cache()
            probs = F.softmax(output, dim=-1)
            pred = probs.argmax(dim=-1)
            correct += pred.eq(target).sum().item()
            
            metrics.update(probs, target)
            confusion_matrix.update(pred, target)
            
            del target
            gc.collect()
            torch.cuda.empty_cache()

            pbar.update(self.pbar_update)
        print(f'Correct: {correct}/{total} ({correct/total:.4f})')
        return metrics, confusion_matrix

    def fit(self, n_epoch=10):
        device = self.device
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=0.1,
            steps_per_epoch=len(self.train_loader), 
            epochs=n_epoch)
        self.model.to(device)
        with tqdm(total=n_epoch) as pbar:
            for epoch in range(1, n_epoch + 1):
                self.train_step(epoch, pbar)
                acc = self.test_step(epoch)
                self.test_accuracy += [acc]
                self.scheduler.step()
        self.model.to('cpu')
        

    def save(self, checkpoint_path):
        torch.save(self.model.state_dict(), checkpoint_path)
        self.model.load_state_dict(torch.load(checkpoint_path))


def get_dataloader(config, data_dir):
    train_loader, test_loader = None, None
    return train_loader, test_loader


def main(config, data_dir, checkpoint_path):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    trainer = Trainer(config)
    train_loader, test_loader = get_dataloader(config, data_dir)
    trainer.prepare(train_loader, test_loader)
    trainer.setup()
    trainer.fit()
    
    metrics, confusion_matrix = trainer.evaluate()
    pprint(metrics.compute())
    confusion_matrix.compute()

    trainer.save(checkpoint_path)



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
        '--checkpoint_dir', type=str,
        default='checkpoints/',
        help='Model checkpoints directory')
    args = parser.parse_args()

    main(args.config, args.data_dir, args.checkpoint_path)







