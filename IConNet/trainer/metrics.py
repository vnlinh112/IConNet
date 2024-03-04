from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassPrecision, 
    MulticlassRecall, MulticlassF1Score, 
    MulticlassConfusionMatrix,
    MulticlassAUROC
)
from torchmetrics import MetricCollection
from typing import Literal

def get_metrics(
        num_classes: int,
        step: Literal['train', 'val', 'test']='train'):
    n = num_classes
    if step == 'train':
        metrics = MetricCollection({
            'acc': MulticlassAccuracy(num_classes=n, average='micro'), 
        }, prefix='train_')

    if step == 'val':
        metrics = MetricCollection({
            'acc': MulticlassAccuracy(num_classes=n, average='micro'), 
            'UA': MulticlassAccuracy(num_classes=n, average='macro'), 
            'WA': MulticlassAccuracy(num_classes=n, average='weighted'), 
            'UF1': MulticlassF1Score(num_classes=n, average='macro'),  
            'WF1': MulticlassF1Score(num_classes=n, average='weighted'), 
        }, prefix='val_')

    if step == 'test':
        metrics = MetricCollection({
            'acc': MulticlassAccuracy(num_classes=n, average='micro'), 
            'UA': MulticlassAccuracy(num_classes=n, average='macro'), 
            'WA': MulticlassAccuracy(num_classes=n, average='weighted'),        
            'UF1': MulticlassF1Score(num_classes=n, average='macro'),  
            'WF1': MulticlassF1Score(num_classes=n, average='weighted'),  

            'UAR': MulticlassRecall(num_classes=n, average='macro'), 
            'WAP': MulticlassPrecision(num_classes=n, average='weighted'),
            'ROCAUC': MulticlassAUROC(num_classes=n, average='macro', thresholds=None),
        }, prefix='test_')

    return metrics

def get_detail_metrics(
        num_classes: int,
        step: Literal['test']='test'):
    n = num_classes

    metrics = MetricCollection({
        'f1s_detail': MulticlassF1Score(num_classes=n, average=None),  
        'acc_detail': MulticlassAccuracy(num_classes=n, average=None), 
        'precision_detail': MulticlassPrecision(num_classes=n, average=None),
        'recall_detail': MulticlassRecall(num_classes=n, average=None), 
        'rocauc_detail': MulticlassAUROC(num_classes=n, average=None, thresholds=None)
    }, prefix='test_')

    confusion_matrix = MulticlassConfusionMatrix(num_classes=n, prefix='test_')

    return metrics, confusion_matrix