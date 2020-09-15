# coding: utf-8

import warnings
warnings.simplefilter('ignore')

import argparse
import math
from model import *
from decode import *

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default="", help="model name")
args = parser.parse_args()

knowledge_graph = load_knowledge_graph("./data/resource.txt")

torch.backends.cudnn.benchmark = True

device_name = 'cuda:'+str(CUDA) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)

target_dict = create_dictionary("./data/resource.txt")
glove_vectors = load_glove("./data/glove.840B.300d.txt", target_dict)

dialog_corpus = load_dialog_corpus("./data/testset.txt", MAX_TEST_DIALOG_CORPUS_SIZE)

encoder = Encoder().to(device)
decoder = Decoder(target_dict['nword']).to(device)
encoder.load("./model/encoder" + args.model + ".pth", device_name)
decoder.load("./model/decoder" + args.model + ".pth", device_name)
encoder.eval()
decoder.eval()

criterion = nn.NLLLoss()

perplexity = 0
with torch.no_grad():
    for input, output in dialog_corpus:
        input_tensor = batch_to_tensor([input], glove_vectors, device)
        output_source_batch_tensor = batch_to_tensor([['_GO'] + output[:-1]], glove_vectors, device)
        output_target_batch_tensor = batch_to_id_tensor([output[1:] + ['_EOS']], target_dict, device)

        hs, h = encoder(input_tensor, None)

        decoder_output, _, attention_weight = decoder(output_source_batch_tensor, hs, h, None, device)
        decoder_output = F.log_softmax(decoder_output, dim=2)

        H = 0
        for i in range(decoder_output.size()[1]):
            H += criterion(decoder_output[:, i, :], output_target_batch_tensor[:, i]).item()
        H /= decoder_output.size()[1]
        perplexity += math.exp(H)
perplexity /= len(dialog_corpus)

test_log_name = "./log/perplexity" + args.model + ".txt"
with open(test_log_name, 'w', encoding='utf-8') as f:
    f.write(args.model + ": " + str(perplexity) + "\n")
