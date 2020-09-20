# coding: utf-8

from utils import *
from param import *
import math

dialog_corpus = load_dialog_corpus("./data/trainset.txt", MAX_DIALOG_CORPUS_SIZE)
corpus_size = 2 * len(dialog_corpus)

knowledge_graph = load_knowledge_graph("./data/resource.txt")
target_dict = create_dictionary("./data/resource.txt")

df = {}
for post, res in dialog_corpus:
    for line in [post, res]:
        for word in set(line):
            if word in knowledge_graph and word in target_dict['word2idx']:
                add_dict(df, word)

df = [[math.log(corpus_size / freq), word] for word, freq in df.items()]
df = sorted(df, reverse=True)
print("min: {}, max: {}".format(df[-1][0], df[0][0]))

with open("./data/idf.txt", 'w', encoding='utf-8') as f:
    for idf, word in df:
        f.write(str(idf / df[0][0]) + "," + word + "\n")
