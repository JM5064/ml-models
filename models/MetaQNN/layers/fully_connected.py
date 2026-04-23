import torch.nn as nn


class FullyConnected(nn.Module):

    def __init__(self, in_features, num_neurons):
        super().__init__()

        self.fc = nn.Linear(in_features=in_features, out_features=num_neurons)


    def forward(self, x):
        x = self.fc(x)

        return x

