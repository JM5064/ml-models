import torch.nn as nn
from ..multiheaded_attention import MultiheadedAttention
from ..mlp import MLP


class EncoderBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.multiheaded_attention = MultiheadedAttention()
        self.mlp = MLP()

    
    def forward(self, X):
        X = self.multiheaded_attention(X, X)
        X = self.mlp(X)

        return X