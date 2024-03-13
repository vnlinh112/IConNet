import torch
from ..nn.model import (
    M13, M13sinc, M13mfcc, M13mel,
    M18, #M18sinc, 
    M18mfcc,
    M19,
    M20, M21, M22)
from ..nn.crnn import CRNN, MFCC_CRNN
from ..visualizer import display_module

MODEL_PICKER = {
            'M13':      M13,
            'M13sinc':  M13sinc,
            'M13mfcc':  M13mfcc,
            'M13mfcc':  M13mfcc,
            'M13mel':   M13mel,
            'M18':      M18,
            # 'M18sinc':  M18sinc,
            'M18mfcc':  M18mfcc,
            'M19':      M19,
            'CRNN':     CRNN,
            'MFCC_CRNN': MFCC_CRNN,
            'M20':      M20,
            'M21':      M21,
            'M22':      M22
        }

class ModelWrapper:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self.pick_model(model_name)
        self.device = 'cpu'

    @staticmethod
    def pick_model(model_name):
        return MODEL_PICKER[model_name]

    def init_model(self, config, n_input, n_output):
        self.config = config
        self.n_input = n_input
        self.n_output = n_output
        self.model = self.model(
            config, n_input=n_input, n_output=n_output)
        return self.model

    def load_from_checkpoint(self, ckpt_path):
        self.ckpt_path = ckpt_path
        self.model = self.model.load_state_dict(
            torch.load(ckpt_path)) 
        return self.model
    
    def set_device(self, device):
        self.device = device
        self.model = self.model.to(device)

    def display_model(
            self, classes_to_visit={}, 
            save_to_path=None):
        input_shape = (1, self.n_input, 16000)
        input = torch.rand(input_shape).to(self.device)
        display_module(
            self.model, input, 
            classes_to_visit=classes_to_visit)

    def generate_visualization(
            self, save_to_path):
        pass

    @staticmethod
    def save_ckpt(
            ckpt_path,
            model,
            optimizer,
            epoch,
            train_losses,
            test_accuracy, 
            log_interval,
            best_epoch=None,
            best_accuracy=None,
            ):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_losses[-1],
            'best_epoch': best_epoch,
            'best_accuracy': best_accuracy,
            'train_losses': train_losses,
            'test_accuracy': test_accuracy,
            'log_interval': log_interval,
            }, ckpt_path)
    
