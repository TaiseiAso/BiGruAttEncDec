# coding: utf-8
# To use this, create folder img/ in ./log/

import warnings
warnings.simplefilter('ignore')

from model import *
from utils import *
from decode import entity_enhance, repetitive_suppression, inf_suppression
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default="", help="model name")
args = parser.parse_args()


def get_near(tokens, near_entities_dict):
    near = [0] * len(tokens)
    for i, token in enumerate(tokens):
        for near_entities in near_entities_dict.values():
            max_n = len(near_entities)
            for n, entities in enumerate(near_entities):
                if token in entities:
                    near[i] = max(near[i], max_n - n)
                    break
    return near


def visualize_greedy_search(decoder, hs, h, glove, dict, device, rep_sup=0.0,
                            graph=None, idf=None, post=None, n=-1, enh=0.0,
                            kg_post=False, kg_res=False, ngram2freq=None, inf_lambda=0.0,
                            kg_enh=False):
    res = ['_GO']
    res_rep_dict = {}
    near_entities_dict = {}
    if kg_post:
        for word in post: add_near_entities_dict(near_entities_dict, word, graph, n)
    topvs = []
    topts = []
    nears = []
    for _ in range(MAX_TEST_LENGTH):
        source_tensor = batch_to_tensor([[res[-1]]], glove, device)
        out, h, _ = decoder(source_tensor, hs, h, None, device)
        out = out[0, 0]
        repetitive_suppression(out, dict, res_rep_dict, rep_sup)
        if kg_enh: entity_enhance(out, dict, near_entities_dict, idf, enh)
        inf_suppression(out, dict, ngram2freq, res, inf_lambda)
        if IGNORE_UNK: out[1] = float('-inf')
        out = F.softmax(out, dim=0)
        topv, topi = out.topk(MAX_VISUALIZE_WIDTH)
        topv, topi = topv.tolist(), topi.tolist()
        topt = [dict['idx2word'][i] for i in topi]
        topvs.append(topv)
        topts.append(topt)
        near = get_near(topt, near_entities_dict)
        nears.append(near)
        token = topt[0]
        if token == '_EOS': break
        add_dict(res_rep_dict, token)
        res.append(token)
        if kg_res:
            add_near_entities_dict(near_entities_dict, token, graph, n)
    return res[1:], topvs, topts, nears


def draw(name, topvs, topts, nears):
    plt.figure(figsize=(1.5*(len(topvs)), 6))
    if VISUALIZE_LOG: plt.yscale("log")
    plt.xlim([0.1, len(topvs)+0.9])
    plt.xlabel("time step", fontsize=15)
    plt.ylabel("probability", fontsize=15)
    plt.xticks(np.arange(1, len(topvs)+1, 1.0))
    vs = [topv[0] for topv in topvs]
    plt.plot(np.array(range(1, len(vs)+1)), vs, c='blue')
    t = 0
    for topv, topt, near in zip(topvs, topts, nears):
        t += 1
        i = 0
        for tv, tt, n in zip(topv, topt, near):
            i += 1
            col = ['k', 'yellow', 'orange', 'red'][min(3, n)]
            if n == 0: plt.scatter(t, tv, c=col, s=10)
            else: plt.scatter(t, tv, c=col, s=30)
            if i == 1 or n > 0: plt.annotate(tt, xy=(t, tv), fontsize=15, color='k')
    plt.savefig(name)


knowledge_graph = load_knowledge_graph("./data/resource.txt")
idf = load_idf("./data/idf.txt")

ngram2freq = None
if TEST_INF_N > 0:
    ngram2freq = load_ngram2freq("./data/" + str(TEST_INF_N) + "ngram2freq.txt")

torch.backends.cudnn.benchmark = True

device_name = 'cuda:'+str(CUDA) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)

target_dict = create_dictionary("./data/resource.txt")
glove_vectors = load_glove("./data/glove.840B.300d.txt", target_dict)

visualize_log_name = "./log/img/visualize" + args.model + "_"

dialog_corpus = load_dialog_corpus("./data/trainset.txt", MAX_VISUALIZE_DIALOG_CORPUS_SIZE)

encoder = Encoder().to(device)
decoder = Decoder(target_dict['nword']).to(device)
encoder.load("./model/encoder" + args.model + ".pth", device_name)
decoder.load("./model/decoder" + args.model + ".pth", device_name)
encoder.eval()
decoder.eval()

with torch.no_grad():
    for i, [input, _] in enumerate(dialog_corpus):
        input_tensor = batch_to_tensor([input], glove_vectors, device)
        hs, h = encoder(input_tensor, None)
        _, topvs, topts, nears = visualize_greedy_search(decoder, hs, h, glove_vectors, target_dict, device,
                rep_sup=0.4, graph=knowledge_graph, post=input, n=2,
                kg_post=True, kg_res=True, ngram2freq=ngram2freq, inf_lambda=0.1)
        draw(visualize_log_name + str(i+1) + ".png", topvs, topts, nears)
        _, topvs, topts, nears = visualize_greedy_search(decoder, hs, h, glove_vectors, target_dict, device,
                rep_sup=0.4, graph=knowledge_graph, idf=idf, post=input, n=2, enh=0.1,
                kg_post=True, kg_res=True, ngram2freq=ngram2freq, inf_lambda=0.1,
                kg_enh=True)
        draw(visualize_log_name + str(i+1) + "(KG).png", topvs, topts, nears)
