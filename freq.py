# coding: utf-8
# To use this, create folder analyze/ in ./log/

from utils import *
from param import *
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=str, default="", help="model name")
args = parser.parse_args()

if args.n <= 0: exit()

dialog_corpus = load_dialog_corpus("./data/trainset.txt", MAX_DIALOG_CORPUS_SIZE)

ngram2freq = {}
top = ['_NORM'] * (args.n - 2) + ['_GO'] * min(1, args.n - 1)
for _, response in dialog_corpus:
    response_ = top + response + ['_EOS']
    for i in range(len(response_) - args.n + 1):
        ngram = ' '.join(response_[i: i + args.n])
        add_dict(ngram2freq, ngram)

ngram2freq = [[freq, ngram] for ngram, freq in ngram2freq.items()]
ngram2freq = sorted(ngram2freq)

plt.hist(np.array(ngram2freq)[:, 0], log=True, bins=100, range=(1, 1000))
plt.savefig("./log/analyze/" + str(args.n) + "gram2freq.png")

with open("./data/" + str(args.n) + "gram2freq.txt", 'w', encoding='utf-8') as f:
    for freq, ngram in ngram2freq:
        f.write(str(freq) + "," + ngram + "\n")
