# coding: utf-8

from utils import *
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.nist_score import corpus_nist
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge
import sys


def eval_length(result):
    length_sum = 0
    for res in result:
        length_sum += len(res)
    return length_sum / len(result)


def eval_distinct(result, n):
    ngram_set = set()
    ngram_sum = 0
    for res in result:
        length = len(res) - n + 1
        ngram_sum += length
        for i in range(length):
            ngram_set.add(' '.join(res[i:i+n]))
    return 100 * len(ngram_set) / ngram_sum


def eval_repeat(result):
    repeat_sum = 0
    for res in result:
        word_set = set()
        for word in res:
            if word in word_set:
                repeat_sum += 1
                break
            word_set.add(word)
    return 100 * repeat_sum / len(result)


def eval_bleu(answers, result, n):
    if n == 1: weights = (1, 0, 0, 0)
    else: weights = (0.5, 0.5, 0, 0)
    bleu_sum = 0
    for ans, res in zip(answers, result):
        bleu_sum += sentence_bleu([ans], res, weights=weights)
    return 100 * bleu_sum / len(result)


def eval_entity(posts, result, graph, n):
    entity_score_sum = 0
    for post, res in zip(posts, result):
        near_entities_dict = {}
        for word in post: add_near_entities_dict(near_entities_dict, word, graph, n)
        for word in res:
            min_n = sys.maxsize
            end_flag = False
            for near_entities in near_entities_dict.values():
                for near, entities in enumerate(near_entities):
                    if word in entities:
                        if near < n or near == 0: end_flag = True
                        min_n = min(min_n, near)
                        break
                if end_flag: break
            if min_n == n: entity_score_sum += 1
    return entity_score_sum / len(result)


def eval_rouge(answers, result, name):
    if name not in ['rouge-1', 'rouge-2', 'rouge-l']:
        name = 'rouge-l'
    answers_ = [' '.join(answer) for answer in answers]
    result = [' '.join(res) for res in result]
    rouge = Rouge()
    scores = rouge.get_scores(result, answers_, avg=True)
    return 100 * scores[name]['f']


def eval_nist(answers, result, n=5):
    answers_ = [[answer] for answer in answers]
    scores = corpus_nist(answers_, result, n)
    return scores


def eval_meteor(answers, result):
    meteor_sum = 0
    for ans, res in zip(answers, result):
        ans, res = ' '.join(ans), ' '.join(res)
        if ans == res: meteor = 1.0
        else: meteor = single_meteor_score(ans, res)
        meteor_sum += meteor
    return 100 * meteor_sum / len(result)
