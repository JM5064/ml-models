import torch
import torch.nn as nn
from ..positional_encoding import PositionalEncoding
from .encoder_block import EncoderBlock
from .decoder_block import DecoderBlock
import time


class OptimusPrime(nn.Module):

    def __init__(self, vocab_size, d_model=512):
        super().__init__()

        self.d_model = d_model

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.positional_encoding = PositionalEncoding()
        self.dropout = nn.Dropout(p=0.1)

        self.encoder = nn.ModuleList()
        for _ in range(6):
            self.encoder.append(EncoderBlock())

        self.decoder = nn.ModuleList()
        for _ in range(6): 
            self.decoder.append(DecoderBlock())

        self.unembedding = nn.Linear(in_features=d_model, out_features=vocab_size)
        self.unembedding.weight = self.embedding.weight


    def forward(self, X, Y):
        # Embed input and output sequences
        X = self.embedding(X) * (self.d_model ** 0.5)
        X = self.positional_encoding(X)
        X = self.dropout(X)

        Y = self.embedding(Y) * (self.d_model ** 0.5)
        Y = self.positional_encoding(Y)

        for layer in self.encoder:
            X = layer(X)

        for layer in self.decoder:
            Y = layer(X, Y)

        Y = self.unembedding(Y)

        return Y


if __name__ == "__main__":
    from data.wikitext import WikiText
    from torch.utils.data.dataloader import DataLoader


    wikitext2 = WikiText()

    dl = DataLoader(wikitext2, batch_size=1)
    op = OptimusPrime(vocab_size=wikitext2.get_vocab_size())

    print("Num params:", sum(p.numel() for p in op.parameters()))

    for X, Y in dl:
        s = time.time()
        op(X, Y)
        print((time.time() - s) * 1000)

        break

