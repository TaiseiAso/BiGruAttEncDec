# coding: utf-8

from param import *


def get_INF_weights(target, ngram2freq):
    if not ngram2freq or INF_N <= 0: return [1.0]*(len(target)+1)
    weights = []
    target_ = ['_NORM']*(INF_N-2) + ['_GO']*min(1, INF_N-1) + target + ['_EOS']
    for i in range(0, len(target_)-INF_N+1):
        weight = 1 / (ngram2freq[' '.join(target_[i: i+INF_N])] ** INF_LAMBDA)
        weights.append(weight)
    return weights


def get_entity_weight(entity, near_entities, enh, ignore):
    n = len(near_entities) - 1
    for near, entities in enumerate(near_entities):
        if entity in entities:
            if near <= ignore: break
            return (n + 2 - near) ** enh
    return 1.0


def get_KG_weights(post, res, graph):
    if not graph or not OBJ_KG: return [1.0]*len(res)
    ignore_post = max(-1, OBJ_IGNORE_POST)
    ignore_res = max(-1, OBJ_IGNORE_RES)
    if OBJ_N_POST <= ignore_post and OBJ_N_RES <= ignore_res: return [1.0]*len(res)
    from utils import get_sentence_near_entities, add_word_near_entities
    post_near_entities = get_sentence_near_entities(post, graph, OBJ_N_POST)
    res_near_entities = []
    weights = []
    for i in range(len(res)):
        weight = get_entity_weight(res[i], post_near_entities, OBJ_ENH_POST, ignore_post)
        if i > 0:
            add_word_near_entities(res_near_entities, res[i-1], graph, OBJ_N_RES)
            weight *= get_entity_weight(res[i], res_near_entities, OBJ_ENH_RES, ignore_res)
        weights.append(weight)
    return weights
