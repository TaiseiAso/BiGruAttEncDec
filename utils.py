# coding: utf-8

from object import *
from param import *
import torch
import json
import random
import copy
import numpy as np


def load_dialog_corpus(path, max_size=-1):
    corpus = []
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx == max_size: break
            json_line = json.loads(line)
            corpus.append([json_line['post'], json_line['response']])
    return corpus


def load_glove(path, dict):
    vectors = {}
    with open(path, 'r', encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == MAX_GROVE_TEST: break
            s = line.strip()
            word = s[:s.find(' ')]
            if word in dict['word2idx']:
                vector = s[s.find(' ') + 1:]
                vectors[word] = list(map(float, vector.split()))
    vectors['_NONE'] = np.zeros(GLOVE_SIZE, dtype=np.float32)
    return vectors


def create_dictionary(path):
    word2idx = {'_PAD': 0, '_UNK': 1, '_GO': 2, '_EOS': 3}
    idx2word = {0: '_PAD', 1: '_UNK', 2: '_GO', 3: '_EOS'}
    nword = 4
    with open(path, 'r', encoding='utf-8') as f:
        json_line = json.loads(f.readline())
        vocab_dict = json_line['vocab_dict']
        vocab_dict_sorted = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
        vocab = [tuple[0] for tuple in vocab_dict_sorted]
        if MAX_VOCAB_SIZE >= 0: vocab = vocab[:MAX_VOCAB_SIZE]
        for idx, word in enumerate(vocab):
            word2idx[word] = nword
            idx2word[nword] = word
            nword += 1
    return {'word2idx': word2idx, 'idx2word': idx2word, 'nword': nword}


def create_dialog_buckets(corpus, graph=None):
    ngram2freq = get_ngram_frequency(corpus)
    bucket_cnt = len(BUCKET_SIZE)
    buckets = [[] for _ in range(bucket_cnt)]
    for dialog in corpus:
        source_len = len(dialog[0])
        target_len = len(dialog[1])
        for bucket_id in range(bucket_cnt):
            if source_len <= BUCKET_SIZE[bucket_id][0] and target_len < BUCKET_SIZE[bucket_id][1]:
                weights = get_INF_weights(dialog[1], ngram2freq)
                if graph:
                    weights_ = get_NG_weights(dialog[0], dialog[1], graph)
                    for i in range(len(weights_)):
                        weights[i] *= weights_[i]
                dialog[1].insert(0, '_GO')
                dialog[1].append('_EOS')
                buckets[bucket_id].append([dialog[0], dialog[1], weights])
                break
    return [bucket for bucket in buckets if bucket != []]


def create_dialog_batchs(buckets):
    batchs = []
    for bucket in buckets:
        random.shuffle(bucket)
        bucket_size = len(bucket)
        for i in range(0, bucket_size, BATCH_SIZE):
            input_batch_length, output_batch_length, input_batch, output_batch, weights_batch = [], [], [], [], []
            for input, output, weights in bucket[i : min(i+BATCH_SIZE, bucket_size)]:
                input_, output_ = copy.copy(input), copy.copy(output)
                input_batch_length.append(len(input_))
                output_batch_length.append(len(output_))
                input_batch.append(input_)
                output_batch.append(output_)
                weights_batch.append(weights)

            arg = np.argsort(input_batch_length)[::-1]
            input_batch_length = [input_batch_length[idx] for idx in arg]
            output_batch_length = [output_batch_length[idx] for idx in arg]
            input_batch = [input_batch[idx] for idx in arg]
            output_batch = [output_batch[idx] for idx in arg]
            weights_batch = [weights_batch[idx] for idx in arg]

            max_input_batch_length = max(input_batch_length)
            max_output_batch_length = max(output_batch_length)
            for j in range(len(input_batch_length)):
                input_batch[j].extend(['_PAD'] * (max_input_batch_length - input_batch_length[j]))
                output_batch[j].extend(['_PAD'] * (max_output_batch_length - output_batch_length[j]))
            batchs.append([input_batch_length, output_batch_length, input_batch, output_batch, weights_batch])
    random.shuffle(batchs)
    return batchs


def batch_to_tensor(batch, glove, device):
    batch_tensor = []
    for data in batch:
        batch_tensor.append(
            [
                random.choice(list(glove.values())) if random.random() < RANDOM_SWAP
                else glove.get(word, glove['_NONE'])
                for word in data
            ]
        )
    return torch.FloatTensor(batch_tensor).to(device)


def batch_to_id_tensor(batch, dict, device):
    batch_id_tensor = []
    for data in batch:
        batch_id_tensor.append([dict['word2idx'].get(word, 1) for word in data])
    return torch.LongTensor(batch_id_tensor).to(device)


def load_knowledge_graph(path):
    knowledge_graph = {}
    with open(path, 'r', encoding='utf-8') as f:
        json_line = json.loads(f.readline())
        triples = json_line['csk_triples']
        for triple in triples:
            entities = triple.split(', ')
            if entities[0] in knowledge_graph:
                knowledge_graph[entities[0]].append(entities[2])
            else:
                knowledge_graph[entities[0]] = [entities[2]]
            if entities[2] in knowledge_graph:
                knowledge_graph[entities[2]].append(entities[0])
            else:
                knowledge_graph[entities[2]] = [entities[0]]
    return knowledge_graph


def get_near_entities_from_knowledge_graph(post, graph, n):
    if not (graph and post and n >= 0): return []
    near_entities = []
    entities = set()
    for word in post:
        if word in graph: entities.add(word)
    near_entities.append(entities)
    for _ in range(n):
        bef_entities = entities
        add_entities = set()
        for entity in entities:
            add_entities = add_entities.union(set(graph[entity]))
        entities = entities.union(add_entities)
        near_entities.append(entities.difference(bef_entities))
    return near_entities


def get_ngram_frequency(corpus):
    if INF_N <= 0: return None
    ngram2freq = {}
    top = ['_NORM']*(INF_N-2) + ['_GO']*min(1, INF_N-1)
    for _, response in corpus:
        response_ = top + response + ['_EOS']
        for i in range(0, len(response_)-INF_N+1):
            ngram = ' '.join(response_[i: i+INF_N])
            if ngram in ngram2freq: ngram2freq[ngram] += 1
            else: ngram2freq[ngram] = 1
    return ngram2freq
