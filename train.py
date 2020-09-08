# coding: utf-8

import warnings
warnings.simplefilter('ignore')

from model import *
from utils import *
import argparse
import torch.optim as optim
import os

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default="", help="model name")
args = parser.parse_args()

param_log_name = "./log/param" + args.model + ".txt"
with open(param_log_name, 'w', encoding='utf-8') as f_out,\
        open("param.py", 'r', encoding='utf-8') as f_in:
    _ = f_in.readline()
    line = f_in.readline()
    while line:
        f_out.write(line)
        line = f_in.readline()

knowledge_graph = None
if OBJ_KG:
    knowledge_graph = load_knowledge_graph("./data/resource.txt")

torch.backends.cudnn.benchmark = True

device_name = 'cuda:'+str(CUDA) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)

target_dict = create_dictionary("./data/resource.txt")
glove_vectors = load_glove("./data/glove.840B.300d.txt", target_dict)

train_log_name = "./log/train" + args.model + ".txt"
if os.path.exists(train_log_name):
    os.remove(train_log_name)

dialog_corpus = load_dialog_corpus("./data/trainset.txt", MAX_DIALOG_CORPUS_SIZE)
dialog_buckets = create_dialog_buckets(dialog_corpus, graph=knowledge_graph)

encoder = Encoder().to(device)
decoder = Decoder(target_dict['nword']).to(device)
encoder.train()
decoder.train()

criterion = nn.NLLLoss(ignore_index=0, reduction='none')

encoder_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

try:
    for epoch in range(1, MAX_EPOCH+1):
        epoch_loss = 0
        dialog_batchs = create_dialog_batchs(dialog_buckets)

        for input_batch_length, output_batch_length, input_batch, output_batch, weights_batch in dialog_batchs:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            input_batch_tensor = batch_to_tensor(input_batch, glove_vectors, device)
            np_output_batch = np.array(output_batch)
            output_source_batch_tensor = batch_to_tensor(np_output_batch[:, :-1], glove_vectors, device)
            output_target_batch_tensor = batch_to_id_tensor(np_output_batch[:, 1:], target_dict, device)

            hs, h = encoder(input_batch_tensor, input_batch_length)

            loss = 0
            decoder_output, _, attention_weight = decoder(output_source_batch_tensor, hs, h, input_batch_length, device)
            decoder_output = F.log_softmax(decoder_output, dim=2)
            for i in range(decoder_output.size()[1]):
                add_loss = batch_size = 0
                batch_loss = criterion(decoder_output[:, i, :], output_target_batch_tensor[:, i])
                for j, bl in enumerate(batch_loss):
                    if output_target_batch_tensor[j, i] != 0:
                        add_loss += weights_batch[j][i] * bl
                        batch_size += 1
                loss += add_loss / batch_size
            loss.backward()

            epoch_loss += loss.item()

            encoder_optimizer.step()
            decoder_optimizer.step()

        epoch_loss /= len(dialog_batchs)
        print_str = "Epoch %d: SCE(%.4f)" % (epoch, epoch_loss)
        print(print_str)
        with open(train_log_name, 'a', encoding='utf-8') as f:
            f.write(print_str + "\n")
    print("Finished")
except KeyboardInterrupt:
    print("Interrupted")

encoder.save("./model/encoder" + args.model + ".pth")
decoder.save("./model/decoder" + args.model + ".pth")
