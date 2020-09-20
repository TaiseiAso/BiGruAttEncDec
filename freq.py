# coding: utf-8

from utils import *
from param import *
import numpy as np
import matplotlib.pyplot as plt

if OBJ_INF_N <= 0: exit()

dialog_corpus = load_dialog_corpus("./data/trainset.txt", MAX_DIALOG_CORPUS_SIZE)

ngram2freq = {}
top = ['_NORM'] * (OBJ_INF_N - 2) + ['_GO'] * min(1, OBJ_INF_N - 1)
for _, response in dialog_corpus:
    response_ = top + response + ['_EOS']
    for i in range(len(response_) - OBJ_INF_N + 1):
        ngram = ' '.join(response_[i: i + OBJ_INF_N])
        add_dict(ngram2freq, ngram)

ngram2freq = [[freq, ngram] for ngram, freq in ngram2freq.items()]
ngram2freq = sorted(ngram2freq)

plt.hist(np.array(ngram2freq)[:, 0], log=True, bins=100, range=(1, 1000))
plt.savefig("./log/analyze/" + str(OBJ_INF_N) + "gram2freq.png")

with open("./data/" + str(OBJ_INF_N) + "gram2freq.txt", 'w', encoding='utf-8') as f:
    for freq, ngram in ngram2freq:
        f.write(str(freq) + "," + ngram + "\n")
