from .dataset import PickleDataset
from .waveform import (
    WaveformDataset, 
    Waveform2mfccDataset, 
    SplittedWaveformDataset) 

DATASET_PICKER = {
            'PickleDataset':            PickleDataset,
            'WaveformDataset':          WaveformDataset,
            'Waveform2mfccDataset':     Waveform2mfccDataset,
            'SplittedWaveformDataset':  SplittedWaveformDataset,
        }

class DatasetWrapper:
    def __init__(self, dataset_class):
        self.dataset_class = dataset_class
        self.dataset = self.pick(dataset_class)

    @staticmethod
    def pick(dataset_class):
        return DATASET_PICKER[dataset_class]

    def init(self, config, data_dir, labels):
        self.config = config
        self.dataset = self.dataset(
            config, data_dir=data_dir, labels=labels)
        return self.dataset