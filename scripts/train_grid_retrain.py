"""retrain a model of best config with full training set"""

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


def load_config(config_file):
    args = {}
    with open(config_file, 'rb') as f:
        for line in f:
            if len(line) == 0:
                continue
            param, value = line.strip('\n').split('=')
            args[param] = value
    return args


# how many times to repeat each config. To alleviate optimization variance
n_repeat = 3
# delete previous files in the current model directory?
fresh_start = True
delete_without_prompt = True
setting = sys.argv[1]
dataset = sys.argv[2]
domain = sys.argv[3]
exec_num = sys.argv[4]
data_dir = 'data/' + dataset + '/' + domain
model_dir = ('execs/' + setting + '/' + dataset + '/' +
             domain + '/exec' + str(exec_num))
config_file = os.path.join(model_dir, 'config')
prev_config = load_config(config_file)
best_steps = int(prev_config['best_steps_on_validation'])
model_dir = model_dir + '/retrain'

config = {'data_dir': data_dir,
          'model_dir': model_dir,
          'use_pretraining': prev_config['use_pretraining'] == 'True',
          'pretrained_model_dir': prev_config['pretrained_model_dir'],
          'mode': 'retrain',
          'maximum_steps': best_steps,
          'batch_size': int(prev_config['batch_size']),
          'steps_per_checkpoint': int(prev_config['steps_per_checkpoint']),
          'optimization_algorithm': prev_config['optimization_algorithm'],
          'learning_rate': float(prev_config['learning_rate']),
          'adam_epsilon': float(prev_config['adam_epsilon']),
          'num_layers': int(prev_config['num_layers']),
          'size': int(prev_config['size']),
          'use_attention': prev_config['use_attention'] == 'True',
          'use_lstm': prev_config['use_lstm'] == 'True',
          'input_keep_prob': float(prev_config['input_keep_prob']),
          'output_keep_prob': float(prev_config['output_keep_prob'])
          }
if 'use_word2vec' in prev_config:
    config['use_word2vec'] = prev_config['use_word2vec'] == 'True'
if 'embedding_size' in prev_config:
    config['embedding_size'] = int(prev_config['embedding_size'])
if 'word2vec_normalization' in prev_config:
    config['word2vec_normalization'] = prev_config['word2vec_normalization']
if 'vocab_embedding_file' in prev_config:
    config['vocab_embedding_file'] = prev_config['vocab_embedding_file']
if 'train_word2vec_embedding' in prev_config:
    config['train_word2vec_embedding'] = \
        prev_config['train_word2vec_embedding'] == 'True'


if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if fresh_start:
    if not delete_without_prompt:
        reply = raw_input('Delete directory ' + model_dir +
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

grid_search_result_file = os.path.join(model_dir,
                                       'grid_search.log')
config_eval_ppx = []
config_best_eval_ppx = float('inf')
for _ in range(n_repeat):
    tf.reset_default_graph()
    config_best_eval_ppx, config_eval_ppx_single = \
        parser.train_grid(config, config_best_eval_ppx)
    config_eval_ppx.append(config_eval_ppx_single)
    if config_eval_ppx_single < config_best_eval_ppx:
        config_best_eval_ppx = config_eval_ppx_single
config_eval_ppx = sorted(config_eval_ppx)
config['best_eval_ppx'] = config_best_eval_ppx
config['eval_ppx'] = config_eval_ppx
with open(grid_search_result_file, 'wb') as f_out:
    json.dump(config,
              f_out,
              sort_keys=True,
              indent=4,
              separators=(',', ': '))
