# coding: utf-8
# To use this file, put in above folder.

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default="", help="model name")
args = parser.parse_args()

os.system("python greedy_kg.py -m {} -n _gk -r {}".format(args.model, 1.0))
os.system("python eval.py -t {}_gk".format(args.model))
os.system("python bs_kg.py -m {} -n _bsk -r {} -l {} -b {}".format(args.model, 1.0, 1.0, 5))
os.system("python eval.py -t {}_bsk".format(args.model))
os.system("python mmi_kg.py -m {} -n _mmik -r {} -l {}".format(args.model, 1.0, 0.6))
os.system("python eval.py -t {}_mmik".format(args.model))
os.system("python sampling_kg.py -m {} -n _sk -r {} -t {}".format(args.model, 1.0, 0.6))
os.system("python eval.py -t {}_sk".format(args.model))
os.system("python top_k_sampling_kg.py -m {} -n _tksk -r {} -t {} -k {}".format(args.model, 1.0, 0.6, 32))
os.system("python eval.py -t {}_tksk".format(args.model))
os.system("python top_p_sampling_kg.py -m {} -n _tpsk -r {} -t {} -p {}".format(args.model, 1.0, 0.8, 0.6))
os.system("python eval.py -t {}_tpsk".format(args.model))
