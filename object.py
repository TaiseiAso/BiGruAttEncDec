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


def get_entity_weight(entity, near_entities, n, enh, ignore):
    for near, entities in enumerate(near_entities[ignore+1:]):
        if entity in entities:
            return (n + 1 - near - ignore) ** enh
    return 1.0


def get_NG_weights(post, res, graph):
    if not graph or not OBJ_KG: return [1.0]*len(res)
    ignore_post = max(-1, OBJ_IGNORE_POST)
    ignore_res = max(-1, OBJ_IGNORE_RES)
    if OBJ_N_POST <= ignore_post or OBJ_N_RES <= ignore_res: return [1.0]*len(res)
    from utils import get_near_entities_from_knowledge_graph
    post_near_entities = get_near_entities_from_knowledge_graph(post, graph, OBJ_N_POST)
    weights = []
    for i in range(0, len(res)):
        res_near_entities = get_near_entities_from_knowledge_graph(res[:i], graph, OBJ_N_RES)
        weight = get_entity_weight(res[i], post_near_entities, OBJ_N_POST, OBJ_ENH_POST, ignore_post) * \
            get_entity_weight(res[i], res_near_entities, OBJ_N_RES, OBJ_ENH_RES, ignore_res)
        weights.append(weight)
    return weights
