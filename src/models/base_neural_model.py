import torch
import torch.nn as nn
from ..data.datastruct import Batch

class BaseModule(nn.Module):
    def forward(self, batch: Batch):
        pass