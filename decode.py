# coding: utf-8

import torch.nn.functional as F
from utils import *


def repetitive_suppression(out, dict, res_rep_dict, rep_sup):
    for tok, rep in res_rep_dict.items():
        out[dict['word2idx'][tok]] /= (1 + rep) ** rep_sup


def entity_enhancer(out, dict, near_entities, n, enh, ignore_n):
    ignore_n = max(-1, ignore_n)
    for near, entities in enumerate(near_entities[1+ignore_n:]):
        enhance = (n + 1 - near - ignore_n) ** enh
        for entity in entities:
            if entity in dict['word2idx']:
                out[dict['word2idx'][entity]] *= enhance


def greedy_search(decoder, hs, h, glove, dict, device, rep_sup=0.0,
                  graph=None, post=None, post_n=-1, post_enh=0.0, post_ignore_n=-1, res_n=0, res_enh=0.0, res_ignore_n=0):
    res = ['_GO']
    res_rep_dict = {}
    post_near_entities = get_sentence_near_entities(post, graph, post_n)
    res_near_entities = []
    for _ in range(MAX_TEST_LENGTH):
        source_tensor = batch_to_tensor([[res[-1]]], glove, device)
        out, h, _ = decoder(source_tensor, hs, h, None, device)
        out = out[0, 0].tolist()
        repetitive_suppression(out, dict, res_rep_dict, rep_sup)
        entity_enhancer(out, dict, post_near_entities, post_n, post_enh, post_ignore_n)
        entity_enhancer(out, dict, res_near_entities, res_n, res_enh, res_ignore_n)
        idx = np.argmax(out)
        token = dict['idx2word'][idx]
        if token == '_EOS': break
        if token in res_rep_dict: res_rep_dict[token] += 1
        else: res_rep_dict[token] = 1
        res.append(token)
        add_word_near_entities(res_near_entities, token, graph, res_n)
    return res[1:]


def sampling_search(decoder, hs, h, glove, dict, device, rep_sup=0.0, temp=1.0,
                    graph=None, post=None, post_n=-1, post_enh=0.0, post_ignore_n=-1, res_n=0, res_enh=0.0, res_ignore_n=0):
    res = ['_GO']
    res_rep_dict = {}
    post_near_entities = get_sentence_near_entities(post, graph, post_n)
    res_near_entities = []
    for _ in range(MAX_TEST_LENGTH):
        source_tensor = batch_to_tensor([[res[-1]]], glove, device)
        out, h, _ = decoder(source_tensor, hs, h, None, device)
        out = out[0, 0]
        repetitive_suppression(out, dict, res_rep_dict, rep_sup)
        entity_enhancer(out, dict, post_near_entities, post_n, post_enh, post_ignore_n)
        entity_enhancer(out, dict, res_near_entities, res_n, res_enh, res_ignore_n)
        out = F.softmax(out / temp, dim=0).tolist()
        idx = random.choices(range(len(out)), weights=out)[0]
        token = dict['idx2word'][idx]
        if token == '_EOS': break
        if token in res_rep_dict: res_rep_dict[token] += 1
        else: res_rep_dict[token] = 1
        res.append(token)
        add_word_near_entities(res_near_entities, token, graph, res_n)
    return res[1:]


def top_k_sampling_search(decoder, hs, h, glove, dict, device, rep_sup=0.0, k=1, temp=1.0,
                          graph=None, post=None, post_n=-1, post_enh=0.0, post_ignore_n=-1, res_n=0, res_enh=0.0, res_ignore_n=0):
    res = ['_GO']
    res_rep_dict = {}
    post_near_entities = get_sentence_near_entities(post, graph, post_n)
    res_near_entities = []
    for _ in range(MAX_TEST_LENGTH):
        source_tensor = batch_to_tensor([[res[-1]]], glove, device)
        out, h, _ = decoder(source_tensor, hs, h, None, device)
        out = out[0, 0]
        repetitive_suppression(out, dict, res_rep_dict, rep_sup)
        entity_enhancer(out, dict, post_near_entities, post_n, post_enh, post_ignore_n)
        entity_enhancer(out, dict, res_near_entities, res_n, res_enh, res_ignore_n)
        topv, topi = out.topk(k)
        topv = F.softmax(topv / temp, dim=0)
        idx = random.choices(topi.tolist(), weights=topv.tolist())[0]
        token = dict['idx2word'][idx]
        if token == '_EOS': break
        if token in res_rep_dict: res_rep_dict[token] += 1
        else: res_rep_dict[token] = 1
        res.append(token)
        add_word_near_entities(res_near_entities, token, graph, res_n)
    return res[1:]


def top_p_sampling_search(decoder, hs, h, glove, dict, device, rep_sup=0.0, p=0.0,
                          graph=None, post=None, post_n=-1, post_enh=0.0, post_ignore_n=-1, res_n=0, res_enh=0.0, res_ignore_n=0):
    res = ['_GO']
    res_rep_dict = {}
    post_near_entities = get_sentence_near_entities(post, graph, post_n)
    res_near_entities = []
    for _ in range(MAX_TEST_LENGTH):
        source_tensor = batch_to_tensor([[res[-1]]], glove, device)
        out, h, _ = decoder(source_tensor, hs, h, None, device)
        out = out[0, 0].cpu()
        repetitive_suppression(out, dict, res_rep_dict, rep_sup)
        entity_enhancer(out, dict, post_near_entities, post_n, post_enh, post_ignore_n)
        entity_enhancer(out, dict, res_near_entities, res_n, res_enh, res_ignore_n)
        out = F.softmax(out, dim=0)
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
        if token in res_rep_dict: res_rep_dict[token] += 1
        else: res_rep_dict[token] = 1
        res.append(token)
        add_word_near_entities(res_near_entities, token, graph, res_n)
    return res[1:]


def mmi_antiLM_search(decoder, hs, h, glove, dict, device, rep_sup=0.0, step=0, mmi_lambda=0.0,
                      graph=None, post=None, post_n=-1, post_enh=0.0, post_ignore_n=-1, res_n=0, res_enh=0.0, res_ignore_n=0):
    res = ['_GO']
    res_rep_dict = {}
    post_near_entities = get_sentence_near_entities(post, graph, post_n)
    res_near_entities = []
    hs_mmi = torch.zeros_like(hs)
    h_mmi = torch.zeros_like(h)
    for i in range(MAX_TEST_LENGTH):
        source_tensor = batch_to_tensor([[res[-1]]], glove, device)
        out, h, _ = decoder(source_tensor, hs, h, None, device)
        if step <= 0 or i < step:
            out_mmi, h_mmi, _ = decoder(source_tensor, hs_mmi, h_mmi, None, device)
            out -= mmi_lambda * out_mmi
        out = out[0, 0].tolist()
        repetitive_suppression(out, dict, res_rep_dict, rep_sup)
        entity_enhancer(out, dict, post_near_entities, post_n, post_enh, post_ignore_n)
        entity_enhancer(out, dict, res_near_entities, res_n, res_enh, res_ignore_n)
        idx = np.argmax(out)
        token = dict['idx2word'][idx]
        if token == '_EOS': break
        if token in res_rep_dict: res_rep_dict[token] += 1
        else: res_rep_dict[token] = 1
        res.append(token)
        add_word_near_entities(res_near_entities, token, graph, res_n)
    return res[1:]


def beam_search(decoder, hs, h, glove, dict, device, rep_sup=0.0, B=1, length_norm=1.0, sibling_penalty=0.0,
                graph=None, post=None, post_n=-1, post_enh=0.0, post_ignore_n=-1, res_n=0, res_enh=0.0, res_ignore_n=0):
    return diverse_beam_search(decoder, hs, h, glove, dict, device,
                               rep_sup=rep_sup, B=B, G=1, length_norm=length_norm, sibling_penalty=sibling_penalty, diversity_strength=0,
                               graph=graph, post=post, post_n=post_n, post_enh=post_enh, post_ignore_n=post_ignore_n,
                               res_n=res_n, res_enh=res_enh, res_ignore_n=res_ignore_n)


def diverse_beam_search(decoder, hs, h, glove, dict, device, rep_sup=0.0, B=1, G=1, length_norm=1.0, sibling_penalty=0.0, diversity_strength=0.0,
                        graph=None, post=None, post_n=-1, post_enh=0.0, post_ignore_n=-1, res_n=0, res_enh=0.0, res_ignore_n=0):
    ress = []
    post_near_entities = get_sentence_near_entities(post, graph, post_n)
    beam_sizes = [B for _ in range(G)]
    beams = [[{'res': ['_GO'], 'res_rep_dict': {}, 'res_near_entities': [], 'score': 0, 'hidden': h, 'length': 1}] for _ in range(G)]
    for _ in range(MAX_TEST_LENGTH):
        for g in range(G):
            if beam_sizes[g] == 0: continue
            next_beams = []
            for beam in beams[g]:
                source_tensor = batch_to_tensor([[beam['res'][-1]]], glove, device)
                out, next_h, _ = decoder(source_tensor, hs, beam['hidden'], None, device)
                out = out[0, 0].cpu()
                repetitive_suppression(out, dict, beam['res_rep_dict'], rep_sup)
                entity_enhancer(out, dict, post_near_entities, post_n, post_enh, post_ignore_n)
                entity_enhancer(out, dict, beam['res_near_entities'], res_n, res_enh, res_ignore_n)
                out = F.log_softmax(out, dim=0) + beam['score']
                ranks = out / (beam['length'] ** length_norm)
                out = out.tolist()
                for h in range(g):
                    if beam_sizes[h] == 0: continue
                    for beam_ in beams[h]:
                        idx = dict['word2idx'][beam_['res'][-1]]
                        ranks[idx] -= diversity_strength
                arg = np.argsort(ranks).tolist()[::-1][:beam_sizes[g]]
                for child, idx in enumerate(arg):
                    token = dict['idx2word'][idx]
                    next_res_rep_dict = copy.copy(beam['res_rep_dict'])
                    if token in next_res_rep_dict:
                        next_res_rep_dict[token] += 1
                    else:
                        next_res_rep_dict[token] = 1
                    next_res_near_entities = copy.copy(beam['res_near_entities'])
                    add_word_near_entities(next_res_near_entities, token, graph, res_n)
                    next_beams.append({
                        'res': beam['res']+[token] if beam['length'] < MAX_TEST_LENGTH else beam['res']+[token]+['_EOS'],
                        'res_rep_dict': next_res_rep_dict,
                        'res_near_entities': next_res_near_entities, 'score': out[idx],
                        'hidden': next_h, 'length': beam['length']+1, 'penalty': child * sibling_penalty
                    })
            next_beams = sorted(next_beams, key=lambda x: x['score']-x['penalty'], reverse=True)[:beam_sizes[g]]
            beams[g] = []
            for next_beam in next_beams:
                if next_beam['res'][-1] == '_EOS':
                    beam_sizes[g] -= 1
                    ress.append({'res': next_beam['res'][1:-1], 'score': next_beam['score']/((next_beam['length']-1) ** length_norm)})
                else:
                    del next_beam['penalty']
                    beams[g].append(next_beam)
        if len(ress) == B*G: break
    ress = sorted(ress, key=lambda x: x['score'], reverse=True)
    return ress


def reranking(ress, graph=None):
    if not graph: return ress[0]['res']
    return ress
