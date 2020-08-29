# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, layer, dropout=0):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(embedding_dim, hidden_dim, layer, batch_first=True, dropout=dropout, bidirectional=True)

    def forward(self, embeddings, lengths):
        if lengths:
            embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, lengths=lengths, batch_first=True)
        hs, h = self.gru(embeddings)
        if lengths:
            hs, _ = nn.utils.rnn.pad_packed_sequence(hs, batch_first=True, total_length=lengths[0])
        h = torch.chunk(h, 2, dim=0)
        h = torch.cat([h[0], h[1]], dim=2)
        return hs, h

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device_name):
        if device_name == 'cpu':
            self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(path))


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer, dropout=0):
        super(Decoder, self).__init__()
        self.gru = nn.GRU(embedding_dim, hidden_dim, layer, batch_first=True, dropout=dropout)
        self.concat = nn.Linear(hidden_dim * 2, hidden_dim)
        self.hidden2linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, embeddings, hs, h, mask, device):
        output, h = self.gru(embeddings, h)
        t_output = torch.transpose(output, 1, 2)
        s = torch.bmm(hs, t_output)
        if mask:
            for i in range(s.size()[0]):
                for j in range(mask[i], s.size()[1]):
                    for k in range(s.size()[2]):
                        s[i, j, k] = float('-inf')
        attention_weight = F.softmax(s, dim=1)
        c = torch.zeros(hs.size()[0], 1, hs.size()[2]).to(device)
        for i in range(attention_weight.size()[2]):
            unsq_weight = attention_weight[:, :, i].unsqueeze(2)
            weighted_hs = hs * unsq_weight
            weight_sum = torch.sum(weighted_hs, dim=1).unsqueeze(1)
            c = torch.cat([c, weight_sum], dim=1)
        c = c[:, 1:, :]
        output = torch.cat([output, c], dim=2)
        output = torch.tanh(self.concat(output))
        output = self.hidden2linear(output)
        return output, h, attention_weight

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device_name):
        if device_name == 'cpu':
            self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(path))
