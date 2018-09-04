
#! /use/bin/env python

"""Train model with grid search. Unit variance standardization for word embed"""

from __future__ import absolute_import

import os
import shutil
import sys
import tensorflow as tf
import itertools
import json

lib_path = os.path.abspath('src')
sys.path = [lib_path] + sys.path
import parser

# how many times to repeat each config. alleviate training variance
n_repeat = 3
# delete previous files in the current model directory?
fresh_start = True
# delete without asking?
delete_without_prompt = True
# in-domain or cross-domain?
setting = sys.argv[1]
# overnight or ?
dataset = sys.argv[2]
# a domain in a dataset, e.g., basketball in overnight
domain = sys.argv[3]
# distinctive id for this execution
exec_num = sys.argv[4]
# where to find data
data_dir = 'data/' + dataset + '/' + domain
# where to store trained model
model_dir = ('execs/' + setting + '/' + dataset + '/' +
             domain + '/exec' + str(exec_num))
# for cross-domain setting, use pre-training and specify pretrained model
use_pretraining = False
pretrained_model_dir = 'None'
if len(sys.argv) > 5:   # only when cross-domain
    use_pretraining = True
    pretrain_domain = sys.argv[5]
    pretrain_exec_num = sys.argv[6]
    pretrained_model_dir = ('execs/cross-domain/' + dataset + '/' +
                            pretrain_domain + '/exec' + str(pretrain_exec_num))

all_configs = []    # all configs to do grid search over
shared_configs = {'data_dir': data_dir,
                  'model_dir': model_dir,
                  'use_pretraining': use_pretraining,
                  'pretrained_model_dir': pretrained_model_dir,
                  'steps_per_checkpoint': 10,
                  'early_stop_tolerance': 15,
                  'write_summary': False}
all_configs.append([shared_configs])

model_type_configs = [
    {'use_attention': True},
    # {'use_attention': False},
]
all_configs.append(model_type_configs)

batch_size_configs = [
    {'batch_size': 512},
    # {'batch_size': 256},
    # {'batch_size': 128},
]
all_configs.append(batch_size_configs)

opt_algm_configs = [
    {'optimization_algorithm': 'adam',
     'learning_rate': 0.001,
     'adam_epsilon': 1e-8},
    # {'optimization_algorithm': 'adam',
    #  'learning_rate': 0.001,
    #  'adam_epsilon': 1e-7},
    # {'optimization_algorithm': 'adam',
    #  'learning_rate': 0.001,
    #  'adam_epsilon': 1e-6},
    # {'optimization_algorithm': 'rmsprop',
    #  'learning_rate': 0.001},
    # {'optimization_algorithm': 'rmsprop',
    #  'learning_rate': 0.01},
    # {'optimization_algorithm': 'rmsprop',
    #  'learning_rate': 0.0001},
    # {'optimization_algorithm': 'adagrad',
    #  'learning_rate': 0.1},
    # {'optimization_algorithm': 'adagrad',
    #  'learning_rate': 0.05},
    # {'optimization_algorithm': 'adagrad',
    #  'learning_rate': 0.01},
    # {'optimization_algorithm': 'adagrad',
    #  'learning_rate': 0.005},
    # {'optimization_algorithm': 'vanilla',
    #  'learning_rate': 1.0,
    #  'learning_rate_decay_factor': 0.95},
    # {'optimization_algorithm': 'vanilla',
    #  'learning_rate': 0.5,
    #  'learning_rate_decay_factor': 0.95},
    # {'optimization_algorithm': 'vanilla',
    #  'learning_rate': 0.1,
    #  'learning_rate_decay_factor': 0.95},
    # {'optimization_algorithm': 'vanilla',
    #  'learning_rate': 0.05,
    #  'learning_rate_decay_factor': 0.95},
    # {'optimization_algorithm': 'vanilla',
    #  'learning_rate': 0.01,
    #  'learning_rate_decay_factor': 0.95},
    # {'optimization_algorithm': 'adadelta'},
]
all_configs.append(opt_algm_configs)

num_layer_configs = [
    {'num_layers': 1},
    # {'num_layers': 2}
]
all_configs.append(num_layer_configs)

state_size_configs = [
    {'size': 100},          # promising one first to save time
    # {'size': 500},
    # {'size': 800},
    # {'size': 300},
    # {'size': 200},
    # {'size': 50},
]
all_configs.append(state_size_configs)

# reverse_input_configs = [{'reverse_input': False},
#                          # {'reverse_input': True}
#                          ]
# all_configs.append(reverse_input_configs)

cell_type_configs = [
    {'use_lstm': False},
    # {'use_lstm': True}
]
all_configs.append(cell_type_configs)

input_keep_prob_configs = [
    # {'input_keep_prob': 0.9},
    # {'input_keep_prob': 0.8},
    {'input_keep_prob': 0.7},
    # {'input_keep_prob': 0.5},
    # {'input_keep_prob': 1.0},
    # {'input_keep_prob': 0.3},
    # {'input_keep_prob': 0.1},
]
all_configs.append(input_keep_prob_configs)

output_keep_prob_configs = [
    {'output_keep_prob': 0.5},
    # {'output_keep_prob': 0.7},
    # {'output_keep_prob': 0.9},
    # {'output_keep_prob': 1.0},
]
all_configs.append(output_keep_prob_configs)

embedding_configs = [
    {'use_word2vec': True,
     'embedding_size': 300,
     'word2vec_normalization': 'unit_var',
     'vocab_embedding_file': 'vocab.word2vec.unit_var.npy',
     'train_word2vec_embedding': True},
    # {'use_word2vec': True,
    #  'embedding_size': 300,
    #  'word2vec_normalization': 'unit_var',
    #  'vocab_embedding_file': 'vocab.word2vec.unit_var.npy',
    #  'train_word2vec_embedding': False},
    # {'use_word2vec': True,
    #  'embedding_size': 300,
    #  'word2vec_normalization': 'unit_norm',
    #  'vocab_embedding_file': 'vocab.word2vec.unit_norm.npy',
    #  'train_word2vec_embedding': True},
    # {'use_word2vec': True,
    #  'embedding_size': 300,
    #  'word2vec_normalization': 'unit_norm',
    #  'vocab_embedding_file': 'vocab.word2vec.unit_norm.npy',
    #  'train_word2vec_embedding': False},
    # {'use_word2vec': True,
    #  'embedding_size': 300,
    #  'vocab_embedding_file': 'vocab.word2vec.npy',
    #  'train_word2vec_embedding': True},
    # {'use_word2vec': True,
    #  'embedding_size': 300,
    #  'vocab_embedding_file': 'vocab.word2vec.npy',
    #  'train_word2vec_embedding': False},
    # {'use_word2vec': True,
    #  'embedding_size': 300,
    #  'word2vec_normalization': 'feature_unit_var',
    #  'vocab_embedding_file': \
    #      'vocab.word2vec.feature_unit_var.npy',
    #  'train_word2vec_embedding': True},
    # {'use_word2vec': False,
    #  'embedding_size': -1},
    # {'use_word2vec': False,
    #  'embedding_size': 300},
    # {'use_word2vec': False,
    #  'embedding_size': 100},
    # {'use_word2vec': False,
    #  'embedding_size': 50},
]
all_configs.append(embedding_configs)

if not os.path.exists(shared_configs['model_dir']):
    os.makedirs(shared_configs['model_dir'])
if fresh_start:     # delete existing results if fresh start
    if not delete_without_prompt:
        reply = input('Delete directory ' + model_dir +
                      '? [y/[n]] ')
        if reply != 'y':
            print('Aborting...'
                  'Set fresh_start=False to continue interrupted training')
            sys.exit()
    for f in os.listdir(model_dir):
        path = os.path.join(model_dir, f)
        try:
            if os.path.isfile(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            raise e

# save grid search configs
grid_search_config_file = os.path.join(shared_configs['model_dir'],
                                       'grid_search.configs')
with open(grid_search_config_file, 'wb') as f:
    json.dump(all_configs,
              f,
              sort_keys=True,
              indent=4,
              separators=(',', ': '))

grid_search_result_file = os.path.join(shared_configs['model_dir'],
                                       'grid_search.log')
config_result = []
global_best_eval_ppx = float('inf')
for config_tuple in itertools.product(*all_configs):    # iterate over configs
    config = {}
    for config_elem in config_tuple:
        config.update(config_elem)
    config_best_eval_ppx = float('inf')
    config_eval_ppx = []
    for _ in range(n_repeat):   # repeat a config and keep the best
        tf.reset_default_graph()
        global_best_eval_ppx, config_eval_ppx_single = \
            parser.train_grid(config, global_best_eval_ppx)
        config_eval_ppx.append(config_eval_ppx_single)
        if config_eval_ppx_single < config_best_eval_ppx:
            config_best_eval_ppx = config_eval_ppx_single
    config_eval_ppx = sorted(config_eval_ppx)
    config['best_eval_ppx'] = config_best_eval_ppx
    config['eval_ppx'] = config_eval_ppx
    config_result.append(config)
    with open(grid_search_result_file, 'wb') as f_out:
        json.dump(config_result,
                  f_out,
                  sort_keys=True,
                  indent=4,
                  separators=(',', ': '))
