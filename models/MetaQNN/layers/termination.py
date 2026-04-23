import torch.nn as nn


class Termination(nn.Module):

    def __init__(self, in_features):
        super().__init__()


        self.classifier = nn.Linear(in_features=in_features, out_features=10)


    def forward(self, x):
        x = self.classifier(x)

        return x

