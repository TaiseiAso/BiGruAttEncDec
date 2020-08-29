# coding: utf-8

import argparse
import torch.optim as optim
import math
from model import *
from decode import *
from param import *

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, help="train or test")
args = parser.parse_args()


torch.backends.cudnn.benchmark = True

device_name = 'cuda:'+str(CUDA) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)

target_dict = create_dictionary("./data/resource.txt", MAX_VOCAB_SIZE)
glove_vectors = load_glove("./data/glove.840B.300d.txt", target_dict)

if args.mode != 'test':
    dialog_buckets = create_dialog_buckets(load_dialog_corpus("./data/trainset.txt", MAX_DIALOG_CORPUS_SIZE), BUCKET_SIZE)

    encoder = Encoder(GLOVE_SIZE, HIDDEN_SIZE, LAYER, DROPOUT).to(device)
    decoder = Decoder(target_dict['nword'], GLOVE_SIZE, HIDDEN_SIZE * 2, LAYER, DROPOUT).to(device)
    encoder.train()
    decoder.train()

    criterion = nn.NLLLoss(ignore_index=0)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

    try:
        for epoch in range(1, MAX_EPOCH+1):
            epoch_loss = 0
            dialog_batchs = create_dialog_batchs(dialog_buckets)

            for input_batch_length, output_batch_length, input_batch, output_batch in dialog_batchs:
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                input_batch_tensor = batch_to_tensor(input_batch, glove_vectors, device)
                np_output_batch = np.array(output_batch)
                output_source_batch_tensor = batch_to_tensor(np_output_batch[:, :-1], glove_vectors, device)
                output_target_batch_tensor = batch_to_id_tensor(np_output_batch[:, 1:], target_dict, device)

                hs, h = encoder(input_batch_tensor, input_batch_length)

                loss = 0
                decoder_output, _, attention_weight = decoder(output_source_batch_tensor, hs, h, input_batch_length, device)
                for i in range(decoder_output.size()[1]):
                    loss += criterion(decoder_output[:, i, :], output_target_batch_tensor[:, i])
                loss.backward()

                epoch_loss += loss.item()

                encoder_optimizer.step()
                decoder_optimizer.step()

            epoch_loss /= len(dialog_batchs)
            perplexity = math.exp(epoch_loss)
            print("Epoch %d: SCE(%.4f), PER(%.4f)" % (epoch, epoch_loss, perplexity))

    except KeyboardInterrupt:
        print("Done")

    encoder.save("./model/encoder.pth")
    decoder.save("./model/decoder.pth")
else:
    dialog_corpus = load_dialog_corpus("./data/testset.txt", MAX_TEST_DIALOG_CORPUS_SIZE)

    encoder = Encoder(GLOVE_SIZE, HIDDEN_SIZE, LAYER, 0).to(device)
    decoder = Decoder(target_dict['nword'], GLOVE_SIZE, HIDDEN_SIZE * 2, LAYER, 0).to(device)
    encoder.load("./model/encoder.pth", device_name)
    decoder.load("./model/decoder.pth", device_name)
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for input, output in dialog_corpus:
            input_tensor = batch_to_tensor([input], glove_vectors, device)
            hs, h = encoder(input_tensor)
            greedy_res = greedy_search(decoder, hs, h, glove_vectors, target_dict, device)
            print("post: ", ' '.join(input))
            print("answer: ", ' '.join(output))
            print("greedy: ", ' '.join(greedy_res))
            print()
