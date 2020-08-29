# coding: utf-8

import torch.nn.functional as F
from utils import *


def greedy_search(decoder, hs, h, glove, dict, device):
    res = []
    source_tensor = batch_to_tensor([['_GO']], glove, device)
    for _ in range(MAX_TEST_LENGTH):
        out, h, _ = decoder(source_tensor, hs, h, None, device)
        idx = np.argmax(out).item()
        token = dict['idx2word'][idx]
        if token == '_EOS': break
        res.append(token)
        source_tensor = batch_to_tensor([[token]], glove, device)
    return res


def sampling_search(decoder, hs, h, glove, dict, device, temp):
    res = []
    source_tensor = batch_to_tensor([['_GO']], glove, device)
    for _ in range(MAX_TEST_LENGTH):
        out, h, _ = decoder(source_tensor, hs, h, None, device)
        out = F.softmax(out[0, 0] / temp, dim=0).tolist()
        idx = random.choices(range(len(out)), weights=out)[0]
        token = dict['idx2word'][idx]
        if token == '_EOS': break
        res.append(token)
        source_tensor = batch_to_tensor([[token]], glove, device)
    return res


def top_k_sampling_search(decoder, hs, h, glove, dict, device, k, temp):
    res = []
    source_tensor = batch_to_tensor([['_GO']], glove, device)
    for _ in range(MAX_TEST_LENGTH):
        out, h, _ = decoder(source_tensor, hs, h, None, device)
        out = F.softmax(out[0, 0] / temp, dim=0)
        topv, topi = out.topk(k)
        idx = random.choices(topi.tolist(), weights=topv.tolist())[0]
        token = dict['idx2word'][idx]
        if token == '_EOS': break
        res.append(token)
        source_tensor = batch_to_tensor([[token]], glove, device)
    return res


def top_p_sampling_search(decoder, hs, h, glove, dict, device, p, temp):
    res = []
    source_tensor = batch_to_tensor([['_GO']], glove, device)
    for _ in range(MAX_TEST_LENGTH):
        out, h, _ = decoder(source_tensor, hs, h, None, device)
        out = F.softmax(out[0, 0] / temp, dim=0)
        topi = np.argsort(out).tolist()[::-1]
        topv = [out[i] for i in topi]
        sumi, sumv = 0, 0
        for v in topv:
            sumv += v
            sumi += 1
            if sumv >= p: break
        idx = random.choices(topi[:sumi], weights=topv[:sumi])[0]
        token = dict['idx2word'][idx]
        if token == '_EOS': break
        res.append(token)
        source_tensor = batch_to_tensor([[token]], glove, device)
    return res


def mmi_antiLM_search(decoder, hs, h, glove, dict, device, lam):
    res = []
    hs_mmi = torch.zeros_like(hs)
    h_mmi = torch.zeros_like(h)
    source_tensor = batch_to_tensor([['_GO']], glove, device)
    for _ in range(MAX_TEST_LENGTH):
        out, h, _ = decoder(source_tensor, hs, h, None, device)
        out_mmi, h_mmi, _ = decoder(source_tensor, hs_mmi, h_mmi, None, device)
        idx = np.argmax(out - lam * out_mmi).item()
        token = dict['idx2word'][idx]
        if token == '_EOS': break
        res.append(token)
        source_tensor = batch_to_tensor([[token]], glove, device)
    return res
