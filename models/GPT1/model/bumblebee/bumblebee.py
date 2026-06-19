import torch.nn as nn
from ..positional_encoding import PositionalEncoding
from .decoder_block import DecoderBlock
import time


class Bumblebee(nn.Module):

    def __init__(self, vocab_size, d_model=512):
        super().__init__()

        self.d_model = d_model

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.positional_encoding = PositionalEncoding()
        self.dropout = nn.Dropout(p=0.1)

        self.decoder = nn.ModuleList()
        for _ in range(12): 
            self.decoder.append(DecoderBlock())

        nn.init.normal_(self.embedding.weight, std=0.02)
        self.unembedding = nn.Linear(in_features=d_model, out_features=vocab_size)
        self.unembedding.weight = self.embedding.weight


    def forward(self, X):
        # Embed input and output sequences
        X = self.embedding(X) * (self.d_model ** 0.5)
        X = self.positional_encoding(X)
        X = self.dropout(X)

        for layer in self.decoder:
            X = layer(X)

        X = self.unembedding(X)

        return X


if __name__ == "__main__":
    from datasets.Wikitext.wikitext import WikiText
    from torch.utils.data.dataloader import DataLoader


    wikitext2 = WikiText()

    dl = DataLoader(wikitext2, batch_size=1)
    bb = Bumblebee(vocab_size=wikitext2.get_vocab_size())

    print("Num params:", sum(p.numel() for p in bb.parameters()))

    for X, Y in dl:
        s = time.time()
        bb(X)
        print((time.time() - s) * 1000)

        break

