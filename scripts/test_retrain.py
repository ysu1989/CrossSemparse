#!/usr/bin/env python

"""Test the re-trained model"""

import subprocess
import os
import sys


def load_config(config_file):
    args = {}
    with open(config_file, 'rb') as f:
        for line in f:
            if len(line) == 0:
                continue
            param, value = line.strip('\n').split('=')
            param = '--' + param
            args[param] = value
    return args


interpreter = ['python']
script = ['src/parser.py']
setting = sys.argv[1]
dataset = sys.argv[2]
domain = sys.argv[3]
exec_num = sys.argv[4]
eval_with_denotation = 'True'
if len(sys.argv) > 5:
    eval_with_denotation = sys.argv[5]
model_dir = ('execs/' + setting + '/' + dataset + '/' +
             domain + '/exec' + str(exec_num)) + '/retrain'
config_file = os.path.join(model_dir, 'config')
args_dict = load_config(config_file)
args_dict['--mode'] = 'test_rank'
args_dict['--fresh_start'] = 'False'
args_dict['--use_pretraining'] = 'False'
args_dict['--eval_with_denotation'] = eval_with_denotation
args = []
for key, value in args_dict.items():
    args = args + [key, value]
command = interpreter + script + args
for arg in sorted(args_dict):
    print(arg, args_dict[arg])
subprocess.call(command)
