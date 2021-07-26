from typing import Dict

import torch
import torch.nn as nn
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(300, hidden_size, num_layers = num_layers, dropout = dropout, batch_first = True, bidirectional = bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.layer_1 = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, num_class)
        self.act_fn = nn.LeakyReLU(0.1)


    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch, h) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        batch = self.embed(batch)
        out, h = self.gru(batch, h)
        out = self.act_fn(out)
        out = self.dropout(out[:, -1, :].view(out.size(0), -1))
        out = self.layer_1(out)
        out = self.act_fn(out)
        out = self.dropout(out)
        out = self.out(out)
        return out, h
    def init_hidden(self, batch_size, device):
        return torch.autograd.Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_size)).to(device)
