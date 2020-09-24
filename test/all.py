# coding: utf-8
# To use this file, put in above folder.

import warnings
warnings.simplefilter('ignore')

from model import *
from decode import *
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default="", help="model name")
parser.add_argument('-n', '--name', type=str, default="", help="test name")
args = parser.parse_args()

torch.backends.cudnn.benchmark = True

device_name = 'cuda:'+str(CUDA) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)

target_dict = create_dictionary("./data/resource.txt")
glove_vectors = load_glove("./data/glove.840B.300d.txt", target_dict)

test_log_name = "./log/test" + args.model + args.name + ".txt"
if os.path.exists(test_log_name):
    os.remove(test_log_name)

dialog_corpus = load_dialog_corpus("./data/testset.txt", MAX_TEST_DIALOG_CORPUS_SIZE)

encoder = Encoder().to(device)
decoder = Decoder(target_dict['nword']).to(device)
encoder.load("./model/encoder" + args.model + ".pth", device_name)
decoder.load("./model/decoder" + args.model + ".pth", device_name)
encoder.eval()
decoder.eval()

rs = 1.0
ln = 1.0
bs_b = 5
s_t = 0.6
tks_t = 0.8
tks_k = 32
tps_p = 0.5

with torch.no_grad():
    for input, output in dialog_corpus:
        with open(test_log_name, 'a', encoding='utf-8') as f:
            input_tensor = batch_to_tensor([input], glove_vectors, device)
            hs, h = encoder(input_tensor, None)

            f.write("post:" + ' '.join(input) + "\n")
            f.write("answer:" + ' '.join(output) + "\n")

            greedy_res = greedy_search(decoder, hs, h, glove_vectors, target_dict, device,
                                       rep_sup=rs)
            beam_ress = beam_search(decoder, hs, h, glove_vectors, target_dict, device,
                                    rep_sup=rs, B=bs_b, length_norm=ln)
            sampling_res = sampling_search(decoder, hs, h, glove_vectors, target_dict, device,
                                           rep_sup=rs, temp=s_t)
            top_k_sampling_res = top_k_sampling_search(decoder, hs, h, glove_vectors, target_dict, device,
                                                       rep_sup=rs, k=tks_k, temp=tks_t)
            top_p_sampling_res = top_p_sampling_search(decoder, hs, h, glove_vectors, target_dict, device,
                                                       rep_sup=rs, p=tps_p)

            f.write("G RS={}:{}\n".format(rs, ' '.join(greedy_res)))
            f.write("BS RS={} B={} LN={}:{}\n".format(rs, bs_b, ln, ' '.join(reranking(beam_ress))))
            f.write("S RS={} T={}:{}\n".format(rs, s_t, ' '.join(sampling_res)))
            f.write("TKS RS={} K={} T={}:{}\n".format(rs, tks_k, tks_t, ' '.join(top_k_sampling_res)))
            f.write("TPS RS={} P={}:{}\n".format(rs, tps_p, ' '.join(top_p_sampling_res)))

            f.write("\n")
