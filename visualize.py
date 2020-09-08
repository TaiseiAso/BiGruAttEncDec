# coding: utf-8

import warnings
warnings.simplefilter('ignore')

import argparse
import matplotlib.pyplot as plt
from model import *
from decode import *

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default="", help="model name")
parser.add_argument('-n', '--name', type=str, default="", help="visualize name")
args = parser.parse_args()


def repetitive_suppression(out, dict, res_rep_dict, rep_sup):
    for tok, rep in res_rep_dict.items():
        out[dict['word2idx'][tok]] /= (1 + rep) ** rep_sup


def entity_enhancer(out, dict, near_entities, enh, ignore_n):
    ignore_n = max(-1, ignore_n)
    n = len(near_entities) - 1
    for near, entities in enumerate(near_entities[1+ignore_n:]):
        enhance = (n + 1 - near - ignore_n) ** enh
        for entity in entities:
            if entity in dict['word2idx']:
                out[dict['word2idx'][entity]] *= enhance


def get_near(near, tokens, near_entities):
    max_n = len(near_entities)
    for i, token in enumerate(tokens):
        for n, entities in enumerate(near_entities):
            if token in entities:
                near[i] = max_n - n
                break


def greedy_search(decoder, hs, h, glove, dict, device, rep_sup=0.0,
                  graph=None, post=None, post_n=-1, post_enh=0.0, post_ignore_n=-1, res_n=0, res_enh=0.0, res_ignore_n=0):
    res = ['_GO']
    res_rep_dict = {}
    post_near_entities = get_sentence_near_entities(post, graph, post_n)
    res_near_entities = []
    topvs = []
    topts = []
    nears = []
    for _ in range(MAX_TEST_LENGTH):
        source_tensor = batch_to_tensor([[res[-1]]], glove, device)
        out, h, _ = decoder(source_tensor, hs, h, None, device)
        out = out[0, 0]
        repetitive_suppression(out, dict, res_rep_dict, rep_sup)
        entity_enhancer(out, dict, post_near_entities, post_enh, post_ignore_n)
        entity_enhancer(out, dict, res_near_entities, res_enh, res_ignore_n)
        out = F.softmax(out, dim=0)
        topv, topi = out.topk(MAX_VISUALIZE_WIDTH)
        topv, topi = topv.tolist(), topi.tolist()
        topt = [dict['idx2word'][i] for i in topi]
        topvs.append(topv)
        topts.append(topt)
        near = [0] * MAX_VISUALIZE_WIDTH
        get_near(near, topt, post_near_entities)
        nears.append(near)
        token = topt[0]
        if token == '_EOS': break
        if token in res_rep_dict: res_rep_dict[token] += 1
        else: res_rep_dict[token] = 1
        res.append(token)
        add_word_near_entities(res_near_entities, token, graph, res_n)
    return res[1:], topvs, topts, nears


def draw(name, topvs, topts, nears):
    plt.figure(figsize=(2*(len(topvs)), 6))
    if VISUALIZE_LOG: plt.yscale("log")
    plt.xlim([0.1, len(topvs)+0.9])
    plt.xlabel("time step", fontsize=15)
    plt.ylabel("probablity", fontsize=15)
    plt.subplots_adjust(left=0.2, right=0.9, bottom=0.1, top=0.95)
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

torch.backends.cudnn.benchmark = True

device_name = 'cuda:'+str(CUDA) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)

target_dict = create_dictionary("./data/resource.txt")
glove_vectors = load_glove("./data/glove.840B.300d.txt", target_dict)

visualize_log_name = "./log/img/visualize" + args.name + "_"

dialog_corpus = load_dialog_corpus("./data/trainset.txt", MAX_VISUALIZE_DIALOG_CORPUS_SIZE)

encoder = Encoder().to(device)
decoder = Decoder(target_dict['nword']).to(device)
encoder.load("./model/encoder" + args.model + ".pth", device_name)
decoder.load("./model/decoder" + args.model + ".pth", device_name)
encoder.eval()
decoder.eval()

with torch.no_grad():
    for i, [input, output] in enumerate(dialog_corpus):
        input_tensor = batch_to_tensor([input], glove_vectors, device)
        hs, h = encoder(input_tensor, None)
        res, topvs, topts, nears = greedy_search(decoder, hs, h, glove_vectors, target_dict, device,
                rep_sup=0.4, graph=knowledge_graph, post=input, post_n=3)
        draw(visualize_log_name + str(i+1) + ".png", topvs, topts, nears)
        res, topvs, topts, nears = greedy_kg_res = greedy_search(decoder, hs, h, glove_vectors, target_dict, device,
                rep_sup=0.4, graph=knowledge_graph, post=input,
                post_n=3, post_enh=0.1, post_ignore_n=-1, res_n=2, res_enh=0.1, res_ignore_n=0)
        draw(visualize_log_name + str(i+1) + "(KG).png", topvs, topts, nears)
