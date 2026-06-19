import torch.nn as nn
from ..multiheaded_attention import MultiheadedAttention
from ..mlp import MLP


class DecoderBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.masked_multiheaded_attention = MultiheadedAttention(use_mask=True)
        self.mlp = MLP()

    
    def forward(self, X):
        X = self.masked_multiheaded_attention(X=X, Y=X)

        X = self.mlp(X)

        return X
