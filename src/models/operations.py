import torch
import torch.nn as nn


class Operation(nn.Module):
    def __init__(self):
        super(Operation, self).__init__()

    def forward(self, *input):
        return NotImplementedError()
    
    
