import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import numpy as np


class LogisticRegression(torch.nn.Module):
    def __init__(self, num_features, num_output):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, num_output)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        return self.softmax(x)
