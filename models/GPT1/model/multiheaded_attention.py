import torch
import torch.nn as nn


class MultiheadedAttention(nn.Module):

    def __init__(self, d_k=64, d_v=64, d_model=512, num_heads=8, max_seq_len=128, use_mask=False):
        super().__init__()

        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.use_mask = use_mask
        
        self.W_Q = nn.Parameter(torch.zeros(d_model, d_k * num_heads))
        self.W_K = nn.Parameter(torch.zeros(d_model, d_k * num_heads))

        self.W_output = nn.Parameter(torch.zeros(d_v * num_heads, d_model)) # value up
        self.W_V = nn.Parameter(torch.zeros(d_model, d_v * num_heads))      # value down
        # (d_k x num_heads x d_model) x 4 params

        self.dropout = nn.Dropout(p=0.1)
        self.attention_dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(d_model)
        # d_model x 2 params

        if self.use_mask:
            mask = torch.zeros((max_seq_len, max_seq_len))
            for i in range(max_seq_len):
                for j in range(i):
                    mask[j, i] = -torch.inf
            
            self.mask = nn.Buffer(mask)

        self.initialize_weights()


    def initialize_weights(self):
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_output)
        nn.init.xavier_uniform_(self.W_V)

    
    def forward(self, X, Y):
        """X : input of size (batch, seq_len, d_model)"""

        seq_len = X.shape[-2]

        X_resid = Y

        X = self.layer_norm(X)
        Y = self.layer_norm(Y)

        # Compute query, key, and value matrices
        Q = Y @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        # Split into multiple heads
        Q = Q.view(-1, seq_len, self.num_heads, self.d_k)
        Q = Q.transpose(1, 2)
        K = K.view(-1, seq_len, self.num_heads, self.d_k)
        K = K.transpose(1, 2)
        V = V.view(-1, seq_len, self.num_heads, self.d_v)
        V = V.transpose(1, 2)

        # Tranpose keys (not including batch dim)
        K_T = K.transpose(-2, -1)

        # Compute attention
        before_softmax = (Q @ K_T) / (self.d_k ** 0.5)

        if self.use_mask:
            before_softmax = before_softmax + self.mask[:seq_len, :seq_len]
        
        after_softmax = torch.softmax(before_softmax, dim=-1)

        attention = self.attention_dropout(after_softmax) @ V

        # Concatenate heads
        attention = attention.transpose(1, 2).reshape(-1, seq_len, self.d_k * self.num_heads)
        attention = attention @ self.W_output

        # Apply dropout
        attention = self.dropout(attention)
    
        # Add residual connection, and layer norm
        X = attention + X_resid

        return X
    

if __name__ == "__main__":
    input_data1 = torch.randn((32, 128, 512))
    input_data2 = torch.randn((32, 128, 512))

    mha = MultiheadedAttention(d_k=64, d_v=64, d_model=512, max_seq_len=128, use_mask=True)

    mha(input_data1, input_data2)
    