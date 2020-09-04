# coding: utf-8

from param import *


def get_INF_weights(target, ngram2freq):
    if not ngram2freq or INF_N <= 0: return [1.0]*len(target)
    weights = []
    target_ = ['_NORM']*(INF_N-2) + ['_GO']*min(1, INF_N-1) + target + ['_EOS']
    for i in range(0, len(target_)-INF_N+1):
        weight = 1 / (ngram2freq[' '.join(target_[i: i+INF_N])] ** INF_LAMBDA)
        weights.append(weight)
    return weights
