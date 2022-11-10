import torch
from avalanche.models.dynamic_modules import (
    MultiTaskModule,
    MultiHeadClassifier,
)


class SimpleSequenceClassifier(torch.nn.Module):
    def __init__(
        self, input_size, hidden_size, n_classes, rnn_layers=1, batch_first=True
    ):
        super().__init__()
        self.batch_first = batch_first
        self.rnn = torch.nn.LSTM(
            input_size,
            hidden_size,
            num_layers=rnn_layers,
            batch_first=batch_first,
        )
        self.classifier = torch.nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1] if self.batch_first else out[-1]
        out = self.classifier(out)
        return out


class MTSimpleSequenceClassifier(MultiTaskModule):
    def __init__(self, input_size, hidden_size, rnn_layers=1, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.rnn = torch.nn.LSTM(
            input_size,
            hidden_size,
            num_layers=rnn_layers,
            batch_first=batch_first,
        )
        self.classifier = MultiHeadClassifier(hidden_size)

    def forward(self, x, task_labels):
        out, _ = self.rnn(x)
        out = out[:, -1] if self.batch_first else out[-1]
        out = self.classifier(out, task_labels)
        return out


__all__ = ["SimpleSequenceClassifier", "MTSimpleSequenceClassifier"]
