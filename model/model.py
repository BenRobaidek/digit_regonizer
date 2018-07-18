import torch
from torch import autograd, nn
import torch.nn.functional as F

class Net(nn.module):
    def __init__(self):
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=4, kernel_size=5, padding=2)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.conv2 = nn.Conv2d(in_channels=4,out_channels=8, kernel_size=5, padding=2)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.lin = nn.Linear(in_features=28*28*8, out_features=10)

    def forward(self, inp):
        y = self.conv1(inp)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool(y)
        return self.lin(y)
