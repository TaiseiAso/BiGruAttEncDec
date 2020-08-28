# coding: utf-8

from utils import *


def greedy_search(decoder, hs, h, glove, dict, device):
    res = []
    output_source_tensor = batch_to_tensor([['_GO']], glove, device)
    for _ in range(MAX_TEST_LENGTH):
        out, h, _ = decoder(output_source_tensor, hs, h, None, device)
        idx = torch.argmax(out[0]).item()
        token = dict['idx2word'][idx]
        if token == '_EOS': break
        res.append(token)
        output_source_tensor = batch_to_tensor([[token]], glove, device)
    return res