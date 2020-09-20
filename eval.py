# coding: utf-8

import warnings
warnings.simplefilter('ignore')

import argparse
import os
from eval_func import *

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test_name', type=str, default="", help="analyze name")
args = parser.parse_args()

knowledge_graph = load_knowledge_graph("./data/resource.txt")

test_log_name = "./log/analyze" + args.test_name + ".txt"
if not os.path.exists(test_log_name):
    print("No analyze log file")
    exit()

eval_log_name = "./log/eval" + args.test_name + ".txt"
if os.path.exists(eval_log_name):
    os.remove(eval_log_name)

posts = []
answers = []
results = {'human': []}
with open(test_log_name, 'r', encoding='utf-8') as f:
    line = f.readline().strip()
    while line:
        _, post = line.split(':', 1)
        posts.append(post.split())
        _, answer = f.readline().strip().split(':', 1)
        answers.append(answer.split())
        results['human'].append(answers[-1])

        line = f.readline().strip()
        while line != "":
            method, result = line.split(':', 1)
            if method in results: results[method].append(result.split())
            else: results[method] = [result.split()]
            line = f.readline().strip()
        line = f.readline().strip()

max_method_len = max([len(method) for method in results.keys()] + [8]) + 1
with open(eval_log_name, 'a', encoding='utf-8') as f:
    f.write("(Method)" + " " * (max_method_len - 8) + ": ")
    f.write("(Length) (DIST-1) (DIST-2) (Repeat) (BLEU-1) (BLEU-2) (Ent.-0) (Ent.-1) (Ent.-2) (ROUGE-1) (ROUGE-2) (ROUGE-l) (NIST-5) (METEOR)\n")
for i, [method, result] in enumerate(results.items()):
    with open(eval_log_name, 'a', encoding='utf-8') as f:
        f.write(method + " " * (max_method_len - len(method)) + ": ")
        f.write("{:7.3f}".format(eval_length(result)) + "  ")
        f.write("{:7.3f}".format(eval_distinct(result, 1)) + "  ")
        f.write("{:7.3f}".format(eval_distinct(result, 2)) + "  ")
        f.write("{:7.3f}".format(eval_repeat(result)) + "  ")
        f.write("{:7.3f}".format(eval_bleu(answers, result, 1)) + "  ")
        f.write("{:7.3f}".format(eval_bleu(answers, result, 2)) + "  ")
        f.write("{:7.3f}".format(eval_entity(posts, result, knowledge_graph, 0)) + "  ")
        f.write("{:7.3f}".format(eval_entity(posts, result, knowledge_graph, 1)) + "  ")
        f.write("{:7.3f}".format(eval_entity(posts, result, knowledge_graph, 2)) + "  ")
        f.write("{:8.3f}".format(eval_rouge(answers, result, 'rouge-1')) + "  ")
        f.write("{:8.3f}".format(eval_rouge(answers, result, 'rouge-2')) + "  ")
        f.write("{:8.3f}".format(eval_rouge(answers, result, 'rouge-l')) + "  ")
        f.write("{:7.3f}".format(eval_nist(answers, result, n=5)) + "  ")
        f.write("{:7.3f}".format(eval_meteor(answers, result)) + "\n")
