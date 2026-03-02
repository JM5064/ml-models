import torch
import torch.nn as nn
import torch.nn.functional as F


class FKModel(nn.Module):

    def __init__(self, input_dim, output_size):
        super().__init__()

        self.fc1 = nn.Linear(in_features=3*input_dim*input_dim, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=output_size)

    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = F.tanh(x)

        return x
    