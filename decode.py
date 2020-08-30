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


def beam_search(decoder, hs, h, glove, dict, device, B, a, gam):
    return diverse_beam_search(decoder, hs, h, glove, dict, device, B, 1, a, gam, 0)


def diverse_beam_search(decoder, hs, h, glove, dict, device, B, G, a, gam, lam):
    ress = []
    beam_sizes = [B] * G
    beams = [[{'res': ['_GO'], 'score': 0, 'hidden': h}] for _ in range(G)]
    for _ in range(MAX_TEST_LENGTH):
        for g in range(G):
            if beam_sizes[g] == 0: continue
            next_beams = []
            for beam in beams[g]:
                source_tensor = batch_to_tensor([beam['res'][-1]], glove, device)
                out, next_h, _ = decoder(source_tensor, hs, beam['hidden'], None, device)
                out = (F.log_softmax(out[0, 0], dim=0) + beam['score']).tolist()
                ranks = [o / (len(beam['res']) ** a) for o in out]
                for h in range(g):
                    if beam_sizes[h] == 0: continue
                    for beam_ in beams[h]:
                        idx = dict['word2idx'][beam_['res'][-1]]
                        ranks[idx] -= lam
                arg = np.argsort(ranks).tolist()[::-1][:beam_sizes[g]]
                for i, idx in enumerate(arg):
                    next_beams.append({'res': beam['res']+[dict['idx2word'][idx]], 'score': out[idx], 'hidden': next_h, 'penalty': i * gam})
            next_beams = sorted(next_beams, key=lambda x: x['score']-x['penalty'], reverse=True)[:beam_sizes[g]]
            beams[g] = []
            for next_beam in next_beams:
                if next_beam['res'][-1] == '_EOS':
                    beam_sizes[g] -= 1
                    ress.append({'res': next_beam['res'][1:-1], 'score': next_beam['score'] / ((len(next_beam['res'])-1) ** a)})
                else:
                    del next_beam['penalty']
                    beams[g].append(next_beam)
        if len(ress) == B*G: break
    ress = sorted(ress, key=lambda x: x['score'], reverse=True)
    return ress[0]['res']
