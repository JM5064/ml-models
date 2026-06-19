import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):

    def __init__(self, label_smoothing=0.1):
        super().__init__()

        self.label_smoothing = label_smoothing
    

    def forward(self, preds, labels):
        """
        preds: [batch, seq_len, vocab_size]
        labels: [batch, seq_len]

        """

        # Get probabilities of the true values in the prediction
        log_probs = torch.log_softmax(preds, dim=-1)

        nll = -log_probs.gather(dim=2, index=labels.unsqueeze(-1)).squeeze(-1)
        smooth = -log_probs.mean(dim=-1)

        loss = (1 - self.label_smoothing) * nll + self.label_smoothing * smooth
        
        return loss.mean()


if __name__ == "__main__":
    preds = torch.tensor([[
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1]
    ]])

    labels = torch.tensor([
        [0, 1]
    ])

    print(preds.shape)
    print(labels.shape)

    loss = CrossEntropyLoss(vocab_size=3)

    print(loss(preds, labels))

