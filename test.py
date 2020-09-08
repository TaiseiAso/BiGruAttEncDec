# coding: utf-8

import warnings
warnings.simplefilter('ignore')

import argparse
import os
from model import *
from decode import *

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default="", help="model name")
parser.add_argument('-n', '--name', type=str, default="", help="test name")
args = parser.parse_args()

knowledge_graph = load_knowledge_graph("./data/resource.txt")

torch.backends.cudnn.benchmark = True

device_name = 'cuda:'+str(CUDA) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)

target_dict = create_dictionary("./data/resource.txt")
glove_vectors = load_glove("./data/glove.840B.300d.txt", target_dict)

test_log_name = "./log/test" + args.name + ".txt"
if os.path.exists(test_log_name):
    os.remove(test_log_name)

dialog_corpus = load_dialog_corpus("./data/testset.txt", MAX_TEST_DIALOG_CORPUS_SIZE)

encoder = Encoder().to(device)
decoder = Decoder(target_dict['nword']).to(device)
encoder.load("./model/encoder" + args.model + ".pth", device_name)
decoder.load("./model/decoder" + args.model + ".pth", device_name)
encoder.eval()
decoder.eval()

with torch.no_grad():
    for input, output in dialog_corpus:
        with open(test_log_name, 'a', encoding='utf-8') as f:
            input_tensor = batch_to_tensor([input], glove_vectors, device)
            hs, h = encoder(input_tensor, None)

            f.write("post:" + ' '.join(input) + "\n")
            f.write("answer:" + ' '.join(output) + "\n")

            greedy_res = greedy_search(decoder, hs, h, glove_vectors, target_dict, device,
                rep_sup=0.4)
            mmi_antiLM_res = mmi_antiLM_search(decoder, hs, h, glove_vectors, target_dict, device,
                rep_sup=0.4, step=5, mmi_lambda=0.2)
            beam_ress = beam_search(decoder, hs, h, glove_vectors, target_dict, device,
                rep_sup=0.4, B=10, length_norm=2.0, sibling_penalty=1.0)
            diverse_beam_ress = diverse_beam_search(decoder, hs, h, glove_vectors, target_dict, device,
                rep_sup=0.4, B=2, G=5, length_norm=2.0, sibling_penalty=1.0, diversity_strength=0.6)
            sampling_res = sampling_search(decoder, hs, h, glove_vectors, target_dict, device,
                rep_sup=0.4, temp=0.4)
            top_k_sampling_res = top_k_sampling_search(decoder, hs, h, glove_vectors, target_dict, device,
                rep_sup=0.4, k=10, temp=0.4)
            top_p_sampling_res = top_p_sampling_search(decoder, hs, h, glove_vectors, target_dict, device,
                rep_sup=0.4, p=0.5)

            greedy_kg_res = greedy_search(decoder, hs, h, glove_vectors, target_dict, device,
                rep_sup=0.4, graph=knowledge_graph, post=input,
                post_n=2, post_enh=0.1, post_ignore_n=-1, res_n=2, res_enh=0.1, res_ignore_n=0)
            mmi_antiLM_kg_res = mmi_antiLM_search(decoder, hs, h, glove_vectors, target_dict, device,
                rep_sup=0.4, step=5, mmi_lambda=0.2, graph=knowledge_graph, post=input,
                post_n=2, post_enh=0.1, post_ignore_n=-1, res_n=2, res_enh=0.1, res_ignore_n=0)
            beam_kg_ress = beam_search(decoder, hs, h, glove_vectors, target_dict, device,
                rep_sup=0.4, B=10, length_norm=2.0, sibling_penalty=1.0, graph=knowledge_graph, post=input,
                post_n=2, post_enh=0.1, post_ignore_n=-1, res_n=2, res_enh=0.1, res_ignore_n=0)
            diverse_beam_kg_ress = diverse_beam_search(decoder, hs, h, glove_vectors, target_dict, device,
                rep_sup=0.4, B=2, G=5, length_norm=2.0, sibling_penalty=1.0, diversity_strength=0.6, graph=knowledge_graph, post=input,
                post_n=2, post_enh=0.1, post_ignore_n=-1, res_n=2, res_enh=0.1, res_ignore_n=0)
            sampling_kg_res = sampling_search(decoder, hs, h, glove_vectors, target_dict, device,
                rep_sup=0.4, temp=0.4, graph=knowledge_graph, post=input,
                post_n=2, post_enh=0.1, post_ignore_n=-1, res_n=2, res_enh=0.1, res_ignore_n=0)
            top_k_sampling_kg_res = top_k_sampling_search(decoder, hs, h, glove_vectors, target_dict, device,
                rep_sup=0.4, k=10, temp=0.4, graph=knowledge_graph, post=input,
                post_n=2, post_enh=0.1, post_ignore_n=-1, res_n=2, res_enh=0.1, res_ignore_n=0)
            top_p_sampling_kg_res = top_p_sampling_search(decoder, hs, h, glove_vectors, target_dict, device,
                rep_sup=0.4, p=0.5, graph=knowledge_graph, post=input,
                post_n=2, post_enh=0.1, post_ignore_n=-1, res_n=2, res_enh=0.1, res_ignore_n=0)

            f.write("greedy:" + ' '.join(greedy_res) + "\n")
            f.write("mmi-antiLM:" + ' '.join(mmi_antiLM_res) + "\n")
            f.write("beam:" + ' '.join(reranking(beam_ress)) + "\n")
            f.write("diverse beam:" + ' '.join(reranking(diverse_beam_ress)) + "\n")
            f.write("sampling:" + ' '.join(sampling_res) + "\n")
            f.write("top-k sampling:" + ' '.join(top_k_sampling_res) + "\n")
            f.write("top-p sampling:" + ' '.join(top_p_sampling_res) + "\n")

            f.write("greedy kg:" + ' '.join(greedy_kg_res) + "\n")
            f.write("mmi-antiLM kg:" + ' '.join(mmi_antiLM_kg_res) + "\n")
            f.write("beam kg:" + ' '.join(reranking(beam_kg_ress)) + "\n")
            f.write("diverse beam kg:" + ' '.join(reranking(diverse_beam_kg_ress)) + "\n")
            f.write("sampling kg:" + ' '.join(sampling_kg_res) + "\n")
            f.write("top-k sampling kg:" + ' '.join(top_k_sampling_kg_res) + "\n")
            f.write("top-p sampling kg:" + ' '.join(top_p_sampling_kg_res) + "\n")
            f.write("\n")
