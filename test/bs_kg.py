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
parser.add_argument('-r', '--rs', type=float, default=0.0, help="repetitive suppression")
parser.add_argument('-l', '--ln', type=float, default=0.0, help="length normalization")
parser.add_argument('-b', '--bs', type=int, default=1, help="beam size")
args = parser.parse_args()

knowledge_graph = load_knowledge_graph("./data/resource.txt")
idf = load_idf("./data/idf.txt")

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

with torch.no_grad():
    for input, output in dialog_corpus:
        with open(test_log_name, 'a', encoding='utf-8') as f:
            input_tensor = batch_to_tensor([input], glove_vectors, device)
            hs, h = encoder(input_tensor, None)

            f.write("post:" + ' '.join(input) + "\n")
            f.write("answer:" + ' '.join(output) + "\n")

            for n in [0, 1, 2]:
                for enh in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                    beam_kg_ress = beam_search(decoder, hs, h, glove_vectors, target_dict, device,
                                               rep_sup=args.rs, B=args.bs, length_norm=args.ln,
                                               graph=knowledge_graph, idf=idf, post=input, n=n, enh=enh,
                                               kg_post=True, kg_res=True)
                    f.write("BSK RS={} B={} LN={} N={} ENH={}:{}\n".format(args.rs, args.bs, args.ln, n, enh, ' '.join(reranking(beam_kg_ress))))

            f.write("\n")
