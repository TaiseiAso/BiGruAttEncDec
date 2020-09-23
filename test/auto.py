# coding: utf-8
# To use this file, put in above folder.

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default="", help="model name")
parser.add_argument('-s', '--step', type=int, default=1, help="experiment step")
parser.add_argument('-r', '--rs', type=str, default="", help="repetitive suppression by step 1")
parser.add_argument('-l', '--ln', type=str, default="", help="length normalization by step 2")
args = parser.parse_args()

if args.step == 1:
    os.system("python greedy.py -m {} -n _g".format(args.model))
    os.system("python eval.py -t {}_g".format(args.model))
elif args.step == 2:
    os.system("python bs.py -m {} -n _bs -r {}".format(args.model, args.rs))
    os.system("python eval.py -t {}_bs".format(args.model))
else:
    os.system("python sibling_bs.py -m {} -n _sbs -r {} -l {}".format(args.model, args.rs, args.ln))
    os.system("python eval.py -t {}_sbs".format(args.model))
    os.system("python group_bs.py -m {} -n _gbs -r {} -l {}".format(args.model, args.rs, args.ln))
    os.system("python eval.py -t {}_gbs".format(args.model))
    os.system("python mmi.py -m {} -n _mmi -r {}".format(args.model, args.rs))
    os.system("python eval.py -t {}_mmi".format(args.model))
    os.system("python sampling.py -m {} -n _s -r {}".format(args.model, args.rs))
    os.system("python eval.py -t {}_s".format(args.model))
    os.system("python top_k_sampling.py -m {} -n _tks -r {}".format(args.model, args.rs))
    os.system("python eval.py -t {}_tks".format(args.model))
    os.system("python top_p_sampling.py -m {} -n _tps -r {}".format(args.model, args.rs))
    os.system("python eval.py -t {}_tps".format(args.model))
