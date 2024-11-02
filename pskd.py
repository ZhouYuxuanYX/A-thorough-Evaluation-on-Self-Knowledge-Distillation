import torch
import math
from torch import nn
from typing import Optional

class DatasetWithIndex(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if isinstance(data, tuple):
            return data + (idx,)
        else:
            return (data, idx)


class PSKDLoss(nn.Module):
    def __init__(self, base_criterion: nn.Module, num_classes: int, total_epochs: int = 300, alpha_T: float = 0.8):
        super().__init__()
        self.base_criterion = base_criterion
        self.num_classes = num_classes
        self.total_epochs = total_epochs
        self.alpha_T = alpha_T
        self.all_predictions = None  # This will be initialized based on the dataset length
        # Note that pskd store all predictions (whole dataset) in cpu.
    
    def set_epoch(self, epoch: int):
        # Calculate alpha_t based on the current epoch
        self.alpha_t = self.alpha_T * ((epoch + 1) / self.total_epochs)
        self.alpha_t = max(0, self.alpha_t)
    
    def initialize_predictions(self, dataset_length: int):
        # Initialize all_predictions tensor with zeros, if not already done
        if self.all_predictions is None:
            self.all_predictions = torch.zeros(dataset_length, self.num_classes)
    
    def forward(self, samples, outputs, targets, input_indices, epoch: int, device):
        if epoch == 0:
            # If it's the first epoch, use hard targets
            soft_targets = targets
        else:
            # Mix the hard targets and soft targets according to alpha_t
            self._check_initialize()
            soft_targets = ((1 - self.alpha_t) * targets) + (self.alpha_t * self.all_predictions[input_indices].to(device))
        
        # Calculate the loss using the base criterion
        loss = self.base_criterion(samples, outputs, soft_targets)
        
        # Update all_predictions with the current outputs
        with torch.no_grad():
            self.all_predictions[input_indices] = torch.nn.functional.softmax(outputs.detach(), dim=1).cpu().float()
        
        return loss

    def _check_initialize(self):
        if self.all_predictions is None:
            raise ValueError("all_predictions is not initialized. Please call initialize_predictions first (e.g. at epoch 0).")