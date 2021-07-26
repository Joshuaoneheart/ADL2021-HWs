from typing import Dict

import torch
import torch.nn as nn
from torch.nn import Embedding
from torchcrf import CRF


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
        self.num_class = num_class
        self.hidden_size = hidden_size
        self.gru = nn.GRU(300, hidden_size, num_layers = num_layers, dropout = dropout, batch_first = True, bidirectional = bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.layer_1 = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size // 2)
        self.out = nn.Linear(hidden_size // 2, num_class)
        self.act_fn = nn.ELU(0.1)
        self.Crf_layer = CRF(num_class, batch_first = True)


    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def argmax(self,vec):
        _, idx = torch.max(vec, 1)
        return idx.item()

    def _viterbi_decode(self, feats):
        seq_len = 40
        batch_size = feats.size(0)
        init_vvars = torch.full((batch_size, seq_len, self.num_class), -100000.).to("cuda:0")

        forward_var = init_vvars
        for i in range(seq_len):
            viterbivars_t = []
            for next_tag in range(self.num_class):
                next_tag_var = forward_var[:, i, :] + self.transitions[next_tag]
                tmp_v, best_tag_id = torch.max(next_tag_var, 1)
                viterbivars_t.append(tmp_v.view(-1, 1))
            if not i:
                tmp_viter = torch.cat(viterbivars_t, 1).view(batch_size, 1, -1)
            else:
                tmp_viter = torch.cat([tmp_viter, torch.cat(viterbivars_t, 1).view(batch_size, 1, -1)], dim = 1)


        forward_var = (tmp_viter + feats).view(batch_size, seq_len, -1)
        
        return forward_var


    def forward(self, batch, labels, h) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        batch = self.embed(batch)
        out, h = self.gru(batch, h)
        out = self.dropout(out)
        out = self.act_fn(out)
        out = self.layer_1(out)
        out = self.dropout(out)
        out = self.act_fn(out)
        out = self.layer_2(out)
        out = self.dropout(out)
        out = self.act_fn(out)
        out = self.out(out)
        
        loss = None
        if labels != None:
            loss = self.Crf_layer(out, labels)

        return loss, self.Crf_layer.decode(out), h
    def init_hidden(self, batch_size, device):
        return torch.autograd.Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_size)).to(device)
