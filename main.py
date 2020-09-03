# coding: utf-8

import warnings
warnings.simplefilter('ignore')

import argparse
import math
import torch.optim as optim
import os
from model import *
from decode import *
from eval import *

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, default="train", help="train or test or eval")
parser.add_argument('-d', '--model_name', type=str, default="", help="model name")
parser.add_argument('-s', '--test_name', type=str, default="", help="test name")
parser.add_argument('-e', '--eval_name', type=str, default="", help="eval name")
args = parser.parse_args()

device_name = device = target_dict = glove_vectors = knowledge_graph = None

if args.mode != 'eval':
    torch.backends.cudnn.benchmark = True

    device_name = 'cuda:'+str(CUDA) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    target_dict = create_dictionary("./data/resource.txt", MAX_VOCAB_SIZE)
    glove_vectors = load_glove("./data/glove.840B.300d.txt", target_dict)

if args.mode != 'train':
    knowledge_graph = load_knowledge_graph("./data/resource.txt")

if args.mode == 'train':
    train_log_name = "./log/train" + args.model_name + ".txt"
    if os.path.exists(train_log_name):
        os.remove(train_log_name)

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
                decoder_output = F.log_softmax(decoder_output, dim=2)
                for i in range(decoder_output.size()[1]):
                    loss += criterion(decoder_output[:, i, :], output_target_batch_tensor[:, i])
                loss.backward()

                epoch_loss += loss.item()

                encoder_optimizer.step()
                decoder_optimizer.step()

            epoch_loss /= len(dialog_batchs)
            perplexity = math.exp(epoch_loss)
            print_str = "Epoch %d: SCE(%.4f), PER(%.4f)" % (epoch, epoch_loss, perplexity)
            print(print_str)
            with open(train_log_name, 'a', encoding='utf-8') as f:
                f.write(print_str + "\n")
    except KeyboardInterrupt:
        print("Done")

    encoder.save("./model/encoder" + args.model_name + ".pth")
    decoder.save("./model/decoder" + args.model_name + ".pth")
elif args.mode == 'test':
    test_log_name = "./log/test" + args.test_name + ".txt"
    if os.path.exists(test_log_name):
        os.remove(test_log_name)

    dialog_corpus = load_dialog_corpus("./data/testset.txt", MAX_TEST_DIALOG_CORPUS_SIZE)

    encoder = Encoder(GLOVE_SIZE, HIDDEN_SIZE, LAYER, 0).to(device)
    decoder = Decoder(target_dict['nword'], GLOVE_SIZE, HIDDEN_SIZE * 2, LAYER, 0).to(device)
    encoder.load("./model/encoder" + args.model_name + ".pth", device_name)
    decoder.load("./model/decoder" + args.model_name + ".pth", device_name)
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
                            rep_sup=0.4, B=10, time_norm=2.0, parent_penalty=1.0)
                diverse_beam_ress = diverse_beam_search(decoder, hs, h, glove_vectors, target_dict, device,
                            rep_sup=0.4, B=2, G=5, time_norm=2.0, parent_penalty=1.0, diverse_penalty=0.6)
                sampling_res = sampling_search(decoder, hs, h, glove_vectors, target_dict, device,
                            rep_sup=0.4, temp=0.4)
                top_k_sampling_res = top_k_sampling_search(decoder, hs, h, glove_vectors, target_dict, device,
                            rep_sup=0.4, k=10, temp=0.4)
                top_p_sampling_res = top_p_sampling_search(decoder, hs, h, glove_vectors, target_dict, device,
                            rep_sup=0.4, p=0.5, temp=0.4)

                greedy_kg_res = greedy_search(decoder, hs, h, glove_vectors, target_dict, device,
                            rep_sup=0.4, graph=knowledge_graph, post=input,
                            post_n=1, post_enh=0.2, post_ignore_n=-1, res_n=1, res_enh=0.2, res_ignore_n=0)
                mmi_antiLM_kg_res = mmi_antiLM_search(decoder, hs, h, glove_vectors, target_dict, device,
                            rep_sup=0.4, step=5, mmi_lambda=0.2, graph=knowledge_graph, post=input,
                            post_n=1, post_enh=0.2, post_ignore_n=-1, res_n=1, res_enh=0.2, res_ignore_n=0)
                beam_kg_ress = beam_search(decoder, hs, h, glove_vectors, target_dict, device,
                            rep_sup=0.4, B=10, time_norm=2.0, parent_penalty=1.0, graph=knowledge_graph, post=input,
                            post_n=1, post_enh=0.2, post_ignore_n=-1, res_n=1, res_enh=0.2, res_ignore_n=0)
                diverse_beam_kg_ress = diverse_beam_search(decoder, hs, h, glove_vectors, target_dict, device,
                            rep_sup=0.4, B=2, G=5, time_norm=2.0, parent_penalty=1.0, diverse_penalty=0.6, graph=knowledge_graph, post=input,
                            post_n=1, post_enh=0.2, post_ignore_n=-1, res_n=1, res_enh=0.2, res_ignore_n=0)
                sampling_kg_res = sampling_search(decoder, hs, h, glove_vectors, target_dict, device,
                            rep_sup=0.4, temp=0.4, graph=knowledge_graph, post=input,
                            post_n=1, post_enh=0.2, post_ignore_n=-1, res_n=1, res_enh=0.2, res_ignore_n=0)
                top_k_sampling_kg_res = top_k_sampling_search(decoder, hs, h, glove_vectors, target_dict, device,
                            rep_sup=0.4, k=10, temp=0.4, graph=knowledge_graph, post=input,
                            post_n=1, post_enh=0.2, post_ignore_n=-1, res_n=1, res_enh=0.2, res_ignore_n=0)
                top_p_sampling_kg_res = top_p_sampling_search(decoder, hs, h, glove_vectors, target_dict, device,
                            rep_sup=0.4, p=0.5, temp=0.4, graph=knowledge_graph, post=input,
                            post_n=1, post_enh=0.2, post_ignore_n=-1, res_n=1, res_enh=0.2, res_ignore_n=0)

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
elif args.mode == 'eval':
    test_log_name = "./log/test" + args.test_name + ".txt"
    if not os.path.exists(test_log_name):
        print("No test log file")
        exit()

    eval_log_name = "./log/eval" + args.eval_name + ".txt"
    if os.path.exists(eval_log_name):
        os.remove(eval_log_name)

    posts = []
    answers = []
    results = {'human': []}
    isFirst = True
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
                if isFirst: results[method] = [result.split()]
                else: results[method].append(result.split())
                line = f.readline().strip()
            line = f.readline().strip()
            isFirst = False

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
