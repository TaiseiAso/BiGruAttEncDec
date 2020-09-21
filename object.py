# coding: utf-8

from param import *


def get_inf_weights(target, ngram2freq):
    if not ngram2freq: return [1.0]*(len(target)+1)
    weights = []
    target_ = ['_NORM'] * (OBJ_INF_N - 2) + ['_GO'] * min(1, OBJ_INF_N - 1) + target + ['_EOS']
    for i in range(0, len(target_) - OBJ_INF_N + 1):
        ngram = ' '.join(target_[i: i + OBJ_INF_N])
        freq = ngram2freq[ngram] if ngram in ngram2freq else 1
        weight = 1 / (freq ** OBJ_INF_LAMBDA)
        weights.append(weight)
    return weights


def get_entity_weight(entity, near_entities_dict, idf):
    max_weight = 1.0
    for word, near_entities in near_entities_dict.items():
        for near, entities in enumerate(near_entities):
            if entity in entities:
                idf_ = idf[word] if idf and word in idf else 1.0
                weight = idf_ * ((OBJ_KG_N + 2 - near) ** OBJ_KG_ENH - 1) + 1
                max_weight = max(max_weight, weight)
                break
    return max_weight


def get_kg_weights(post, res, graph, idf):
    if not graph: return None
    from utils import add_near_entities_dict
    near_entities_dict = {}
    if OBJ_KG_POST:
        for word in post: add_near_entities_dict(near_entities_dict, word, graph, OBJ_KG_N)
    weights = []
    for i in range(len(res)):
        if OBJ_KG_RES and i > 0: add_near_entities_dict(near_entities_dict, res[i-1], graph, OBJ_KG_N)
        weight = get_entity_weight(res[i], near_entities_dict, idf)
        weights.append(weight)
    return weights
