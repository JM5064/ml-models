import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, d_model=512):
        super().__init__()

        self.layer1 = nn.Linear(in_features=d_model, out_features=2048)
        self.layer2 = nn.Linear(in_features=2048, out_features=d_model)

        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(d_model)


    def forward(self, X):
        """X : input of size (batch, seq_len, d_model)"""

        X_resid = X

        X = self.layer_norm(X)

        # Pass through mlp
        X = self.layer1(X)
        X = F.relu(X)
        X = self.layer2(X)

        # Apply dropout
        X = self.dropout(X)

        # Add residual connection and layer norm
        X = X + X_resid

        return X
    

if __name__ == "__main__":
    input_data = torch.randn((16, 128, 512))
    mlp = MLP()

    mlp(input_data)
