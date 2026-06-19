import torch.nn as nn
from model.multiheaded_attention import MultiheadedAttention
from model.mlp import MLP


class DecoderBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.masked_multiheaded_attention = MultiheadedAttention(use_mask=True)
        self.multiheaded_attention = MultiheadedAttention(use_mask=False)
        self.mlp = MLP()

    
    def forward(self, X, Y):
        Y = self.masked_multiheaded_attention(Y, Y)
        X = self.multiheaded_attention(X=X, Y=Y)

        X = self.mlp(X)

        return X
