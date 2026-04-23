import torch.nn as nn
import torch.nn.functional as F

class Convolution(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same')
        

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)

        return x

