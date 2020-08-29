# coding: utf-8

import torch
import json
import random
import numpy as np
from param import *


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


def create_dictionary(path, max_size=-1):
    word2idx = {'_PAD': 0, '_UNK': 1, '_GO': 2, '_EOS': 3}
    idx2word = {0: '_PAD', 1: '_UNK', 2: '_GO', 3: '_EOS'}
    nword = 4
    with open(path, 'r', encoding='utf-8') as f:
        json_line = json.loads(f.readline())
        vocab_dict = json_line['vocab_dict']
        vocab_dict_sorted = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
        vocab = [tuple[0] for tuple in vocab_dict_sorted]
        if max_size >= 0: vocab = vocab[:max_size]
        for idx, word in enumerate(vocab):
            word2idx[word] = nword
            idx2word[nword] = word
            nword += 1
    return {'word2idx': word2idx, 'idx2word': idx2word, 'nword': nword}


def create_dialog_buckets(corpus, bucket_size):
    bucket_cnt = len(bucket_size)
    buckets = [[] for _ in range(bucket_cnt)]
    for dialog in corpus:
        source_len = len(dialog[0])
        target_len = len(dialog[1])
        for bucket_id in range(bucket_cnt):
            if source_len <= bucket_size[bucket_id][0] and target_len < bucket_size[bucket_id][1]:
                dialog[1].insert(0, '_GO')
                dialog[1].append('_EOS')
                buckets[bucket_id].append(dialog)
                break
    return [bucket for bucket in buckets if bucket != []]


def create_dialog_batchs(buckets):
    batchs = []
    for bucket in buckets:
        random.shuffle(bucket)
        bucket_size = len(bucket)
        for i in range(0, bucket_size, BATCH_SIZE):
            input_batch_length, output_batch_length, input_batch, output_batch = [], [], [], []
            for input, output in bucket[i : min(i+BATCH_SIZE, bucket_size)]:
                input_batch_length.append(len(input))
                output_batch_length.append(len(output))
                input_batch.append(input)
                output_batch.append(output)

            arg = np.argsort(input_batch_length)[::-1]
            input_batch_length = [input_batch_length[idx] for idx in arg]
            output_batch_length = [output_batch_length[idx] for idx in arg]
            input_batch = [input_batch[idx] for idx in arg]
            output_batch = [output_batch[idx] for idx in arg]

            max_input_batch_length = input_batch_length[0]
            max_output_batch_length = max(output_batch_length)
            for j in range(len(input_batch_length)):
                input_batch[j].extend(['_PAD'] * (max_input_batch_length - input_batch_length[j]))
                output_batch[j].extend(['_PAD'] * (max_output_batch_length - output_batch_length[j]))
            batchs.append([input_batch_length, output_batch_length, input_batch, output_batch])
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
