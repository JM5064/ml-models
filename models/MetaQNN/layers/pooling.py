import torch.nn as nn


class Pooling(nn.Module):

    def __init__(self, kernel_size, stride):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)


    def forward(self, x):
        x = self.pool(x)

        return x

