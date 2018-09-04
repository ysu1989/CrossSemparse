import os
import sys
import math
import time
import numpy as np
import logging
import shutil
import subprocess
import json

import tensorflow.python.platform
import tensorflow as tf

import utils
import data_utils
import seq2seq_paraphrase_model

# data configuration
tf.app.flags.DEFINE_boolean("fresh_start", False,
                            "True to delete all the files in model_dir")
tf.app.flags.DEFINE_boolean("delete_without_prompt", False,
                            "When fresh_start=True, set this to True if "
                            "we don't want prompt to confirm deletion.")
tf.app.flags.DEFINE_string("data_dir", None, "Data directory")
tf.app.flags.DEFINE_string("test_data_dir", None,
                           "Test data directory for cross-domain evaluation, "
                           "Usually different from data_dir")
tf.app.flags.DEFINE_string("model_dir", "model", "Model directory.")
tf.app.flags.DEFINE_boolean("use_pretraining", False,
                            "True to initialize model with "
                            "pretrained parameters.")
tf.app.flags.DEFINE_string("pretrained_model_dir", None,
                           "If we pretrained a model and want to use that "
                           "for initialization, specify it here.")
tf.app.flags.DEFINE_string("vocab_file", "vocab.txt", "Vocabulary file.")
tf.app.flags.DEFINE_integer("vocab_size", -1, "Vocabulary size.")
tf.app.flags.DEFINE_integer("encoder_size", -1, "Max encoder input length.")
tf.app.flags.DEFINE_integer("decoder_size", -1, "Max decoder input length.")
tf.app.flags.DEFINE_string("vocab_embedding_file", None,
                           "File of vocab embeddings.")
tf.app.flags.DEFINE_string("word2vec_normalization", None,
                           "how to normalize word2vec embeddings. "
                           "{unit_var, unit_norm, None}")
tf.app.flags.DEFINE_string("train_log", None,
                           "Training log file.")
tf.app.flags.DEFINE_string("test_log", None,
                           "Testing log file.")
# learning configuration
tf.app.flags.DEFINE_string("optimization_algorithm", "vanilla",
                           "optimization algorithm: "
                           "{vanilla, adam, adagrad, adadelta, rmsprop}")
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate, only used in "
                          "vanilla, adagrad, and RMSProp.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 1.0,
                          "Learning rate decays by this much, 1.0 = disabled.")
tf.app.flags.DEFINE_float("adam_epsilon", 1e-8,
                          "A small constant for numerical stability in adam.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("maximum_steps", 10000000,
                            "Maximum number of training steps to do.")
tf.app.flags.DEFINE_boolean("write_summary", False,
                            "True if writing summary.")
tf.app.flags.DEFINE_boolean("summarize_trainable_variables", False,
                            "True if summarize trainable variables to view "
                            "in tensorboard (costly).")
tf.app.flags.DEFINE_integer("early_stop_tolerance", 10,
                            "Halt training if no improvement\'s been seen in "
                            "the last n evaluations on the validation set.")
# model configuration
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("size", 100, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size", -1, "Size of word embeddings.")
tf.app.flags.DEFINE_boolean("use_attention", False,
                            "Whether to use the attention mechanism.")
tf.app.flags.DEFINE_boolean("use_lstm", False,
                            "Whether to use the LSTM units. "
                            "The default is GRU units")
tf.app.flags.DEFINE_boolean("use_word2vec", False,
                            "Use word2vec embeddings for initialization?")
tf.app.flags.DEFINE_boolean("train_word2vec_embedding", True,
                            "If use word2vec embeddings, whether train them.")
tf.app.flags.DEFINE_float("input_keep_prob", 1.0,
                          "Dropout: probability to keep input.")
tf.app.flags.DEFINE_float("output_keep_prob", 1.0,
                          "Dropout: probability to keep output.")
# use mode
tf.app.flags.DEFINE_string("mode", "train", "{train, retrain, test_rank}.")
tf.app.flags.DEFINE_boolean("eval_with_denotation", False,
                            "Whether to use denotation instead of canonical "
                            "utterance/logical form to measure prediction "
                            "correctness. This is a plausible strategy, and "
                            "is necessary for the overnight data (employed "
                            "in the original paper). This's because there are "
                            "some logical forms in the overnight data that "
                            "have exactly the same denotation. "
                            "For example, two logical forms might involve the "
                            "same set of facts but are presented in a "
                            "different order.")

FLAGS = tf.app.flags.FLAGS


def read_data(source_path, target_path):
    """Read data from source and target files."""
    logger = logging.getLogger(__name__)
    data_set = []
    max_source_length = 0
    max_target_length = 0
    with open(source_path, 'r') as source_file:
        with open(target_path, 'r') as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target:
                source = source.strip('\n')
                target = target.strip('\n')
                counter += 1
                if counter % 100000 == 0:
                    logger.debug("reading data line %d" % counter)
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_utils.EOS_ID)
                data_set.append([source_ids, target_ids])
                if len(source_ids) > max_source_length:
                    max_source_length = len(source_ids)
                if len(target_ids) > max_target_length:
                    max_target_length = len(target_ids)
                source, target = source_file.readline(), target_file.readline()
    return data_set, max_source_length, max_target_length


def create_model(session,
                 load_existing_model=True,
                 initial_embeddings=None):
    """Create seq2seq model and initialize or load parameters."""
    model = seq2seq_paraphrase_model.Seq2SeqParaphraseModel(
        FLAGS.encoder_size, FLAGS.decoder_size, FLAGS.vocab_size,
        FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm,
        FLAGS.batch_size, FLAGS.optimization_algorithm,
        FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
        use_lstm=FLAGS.use_lstm,
        use_attention=FLAGS.use_attention,
        use_word2vec=FLAGS.use_word2vec,
        embedding_size=FLAGS.embedding_size,
        initial_embeddings=initial_embeddings,
        train_word2vec_embedding=FLAGS.train_word2vec_embedding,
        summarize_trainable_variables=FLAGS.summarize_trainable_variables,
        adam_epsilon=FLAGS.adam_epsilon)
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    logger = logging.getLogger(__name__)
    if FLAGS.use_pretraining:
        # load model parameters from a pretrained model
        ckpt = tf.train.get_checkpoint_state(FLAGS.pretrained_model_dir)
        assert(os.path.exists(ckpt.model_checkpoint_path))
        logger.info("Reading model parameters from pretrained model: %s" %
                    ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        # initialize uninitialized variables
        vars_to_init = [var for var in tf.all_variables()
                        if 'Adam' in var.name]
        logger.debug('variables to init:')
        for var in vars_to_init:
            logger.debug(var.name)
        init_op = tf.initialize_variables(vars_to_init)
        session.run(init_op)
        # reset globle_steps
        model.global_step.assign(0).eval()
    elif (load_existing_model and ckpt and
            os.path.exists(ckpt.model_checkpoint_path)):
        # load model parameters from a trained model in current model dir
        logger.info("Reading model parameters from %s" %
                    ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        # initialize uninitialized variables. This could be used when
        # the pre-training model configuration and the current configuration
        # are not exactly the same
        vars_to_init = [var for var in tf.all_variables()
                        if 'Adam' in var.name]
        logger.debug('variables to init:')
        for var in vars_to_init:
            logger.debug(var.name)
        init_op = tf.initialize_variables(vars_to_init)
        session.run(init_op)
    else:
        # create a fresh model
        logger.info("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    if FLAGS.mode == 'train' or FLAGS.mode == 'retrain':
        # only write summary in training, not testing. Expensive operation.
        if FLAGS.write_summary:
            log_path = os.path.join(FLAGS.model_dir, 'log')
            model.summary_writer = tf.train.SummaryWriter(log_path,
                                                          session.graph)
    return model


def train_grid(config, global_best_eval_ppx):
    """train a model in grid search and return best validation loss"""
    # hyper-parameter configuration and logging setup
    # Because we don't enter from main, we need to manually call
    # FLAGS._parse_flags() in order to set the default flags
    FLAGS._parse_flags()
    # Save the default flags via deep copy, so we can restore later
    default_flags = dict(FLAGS.__dict__['__flags'])
    for key in config:
        if key not in default_flags:
            raise ValueError('key %s not in FLAGS.' % key)
        setattr(FLAGS, key, config[key])
    log_file = (FLAGS.train_log if (FLAGS.train_log is not None and
                                    FLAGS.train_log != 'None')
                else 'train.log')
    utils.setup_logging(log_dir=FLAGS.model_dir,
                        log_file=log_file)
    logger = logging.getLogger(__name__)
    d = FLAGS.__dict__['__flags']
    s = 'Current configuration:\n'
    for flag in sorted(d):
        s += "%s\t%s\n" % (flag, str(d[flag]))
    logger.info(s)

    # data processing, if not done yet
    logger.info('Preparing data in %s' % FLAGS.data_dir)
    source_paths, target_paths, FLAGS.vocab_size = \
        data_utils.prepare_data(FLAGS.data_dir,
                                FLAGS.vocab_file,
                                FLAGS.vocab_size)
    (source_train_path, source_valid_path, source_test_path) = source_paths
    (target_train_path, target_valid_path, target_test_path) = target_paths

    # Read data
    train_set, max_train_source_length, max_train_target_length = \
        read_data(source_train_path, target_train_path)
    valid_set, max_valid_source_length, max_valid_target_length = \
        read_data(source_valid_path, target_valid_path)
    FLAGS.encoder_size = max([max_train_source_length,
                              max_valid_source_length])
    FLAGS.decoder_size = max([max_train_target_length,
                              max_valid_target_length]) + 1      # add 1 for _GO
    if FLAGS.mode == 'retrain':  # when retraining, use full training data
        train_set = train_set + valid_set
    logger.info("Finished loading data.")
    logger.info("vocab size: %d" % FLAGS.vocab_size)
    logger.info("Train size: %d" % len(train_set))
    logger.info("Valid size: %d" % len(valid_set))
    logger.info("encoder size: %d. decoder size: %d." %
                (FLAGS.encoder_size, FLAGS.decoder_size))

    # run model
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # word2vec embedding initialization
        # only do this before training, not testing
        # This will be overwritten if we load parameters from a pre-trained
        # model after model creation
        initial_embeddings = None
        if FLAGS.use_word2vec:
            # read in word2vec embedding matrix
            logger.info('loading vocab embeddings from: %s' %
                        FLAGS.vocab_embedding_file)
            embedding_path = os.path.join(FLAGS.data_dir,
                                          FLAGS.vocab_embedding_file)
            if not os.path.exists(embedding_path):
                # if embedding file not found, create it
                logger.info('Vocab embeddings not found. Creating...')
                command = ['python',
                           '../../shared/word2vec/vocab2vec.py',
                           FLAGS.data_dir,
                           FLAGS.vocab_file,
                           FLAGS.vocab_embedding_file]
                if (FLAGS.word2vec_normalization is not None and
                        FLAGS.word2vec_normalization != "None"):
                    # how to normalize word2vec embeddings
                    command.append(FLAGS.word2vec_normalization)
                subprocess.call(command)
                logger.info('Finished word2vec embedding creation.')
            initial_embeddings = data_utils.read_embeddings(embedding_path)

        # Create model.
        logger.info("Creating %d layers of %d units." %
                    (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess,
                             load_existing_model=False,
                             initial_embeddings=initial_embeddings)

        # Prepare validation data
        (encoder_inputs_valid, decoder_inputs_valid, targets_valid,
         target_weights_valid, encoder_sequence_length_valid,
         decoder_sequence_length_valid) = \
            model.get_batch(valid_set, batch_size=len(valid_set))

        # This is the training loop.
        step_time, loss, gradient_norm = 0.0, 0.0, 0.0
        best_eval_ppx = float('inf')
        current_step = 0
        no_improv_on_valid = 0
        previous_losses = []
        for current_step in range(model.global_step.eval() + 1,
                                  FLAGS.maximum_steps + 1):
            # Get a batch and make a step.
            start_time = time.time()
            (encoder_inputs, decoder_inputs, targets, target_weights,
             encoder_sequence_length, decoder_sequence_length) = \
                model.get_batch(train_set)
            step_gradient_norm, step_loss, _ = \
                model.step(sess,
                           encoder_inputs,
                           decoder_inputs,
                           targets,
                           target_weights,
                           encoder_sequence_length,
                           decoder_sequence_length,
                           input_keep_prob=FLAGS.input_keep_prob,
                           output_keep_prob=FLAGS.output_keep_prob,
                           mode='train')
            step_time += ((time.time() - start_time) /
                          FLAGS.steps_per_checkpoint)
            loss += step_loss / FLAGS.steps_per_checkpoint
            gradient_norm += step_gradient_norm / FLAGS.steps_per_checkpoint

            # Checkpoint
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                logger.info("global step %d learning rate %.4f "
                            "step-time %.4f perplexity %.4f "
                            "gradient norm %.4f" %
                            (model.global_step.eval(),
                             model.learning_rate.eval(),
                             step_time, perplexity, gradient_norm))

                # for vanilla SGD, decrease learning rate if no improvement
                # was seen over last 10 times.
                if FLAGS.optimization_algorithm == 'vanilla':
                    if (len(previous_losses) > 10 and
                            loss > max(previous_losses[-9:])):
                        sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)

                if FLAGS.mode == 'train':
                    # Run evals on validation set and print their perplexity.
                    _, eval_loss, _ = \
                        model.step(sess,
                                   encoder_inputs_valid,
                                   decoder_inputs_valid,
                                   targets_valid,
                                   target_weights_valid,
                                   encoder_sequence_length_valid,
                                   decoder_sequence_length_valid,
                                   input_keep_prob=1.0,
                                   output_keep_prob=1.0,
                                   mode='test_rank')
                    eval_ppx = (math.exp(float(eval_loss))
                                if eval_loss < 300 else float("inf"))
                    logger.info("==validation: perplexity %.4f # samples %d " %
                                (eval_ppx, len(valid_set)))
                    if eval_ppx < best_eval_ppx:
                        # record the number of steps to achieve best evaluation
                        # performance so that later we can retrain using
                        # training + validation with the same number of steps
                        setattr(FLAGS,
                                'best_steps_on_validation',
                                model.global_step.eval())
                        best_eval_ppx = eval_ppx
                        no_improv_on_valid = 0
                    else:
                        no_improv_on_valid += 1
                        if no_improv_on_valid > FLAGS.early_stop_tolerance:
                            logger.info('Halt training because no '
                                        'improvement has been seen in the '
                                        'last %d evaluations.' %
                                        FLAGS.early_stop_tolerance)
                            break
                    if best_eval_ppx < global_best_eval_ppx:
                        # for grid search, only save checkpoint if a new global
                        # validation score is achieved
                        logger.info('Better model found! '
                                    'Previous best eval ppx = %.4f, '
                                    'New best eval ppx = %.4f.' %
                                    (global_best_eval_ppx, best_eval_ppx))
                        checkpoint_path = os.path.join(FLAGS.model_dir,
                                                       "parsing.ckpt")
                        model.saver.save(sess, checkpoint_path)
                        global_best_eval_ppx = best_eval_ppx

                        # record the so-far best configuration
                        config_log_file = os.path.join(FLAGS.model_dir,
                                                       'config')
                        with open(config_log_file, 'w') as f:
                            d = FLAGS.__dict__['__flags']
                            for flag in sorted(d):
                                f.write("%s=%s\n" % (flag, str(d[flag])))
                elif FLAGS.mode == 'retrain':
                    # determine using training loss when re-training with all
                    # training data because now we have no validation set
                    eval_loss = loss
                    eval_ppx = (math.exp(float(eval_loss))
                                if eval_loss < 300 else float("inf"))
                    if eval_ppx < best_eval_ppx:
                        best_eval_ppx = eval_ppx
                    if best_eval_ppx < global_best_eval_ppx:
                        logger.info('Better model found! '
                                    'Previous best eval ppx = %.4f, '
                                    'New best eval ppx = %.4f.' %
                                    (global_best_eval_ppx, best_eval_ppx))
                        # Only save the so-far best model (based on validation
                        # perplexity)
                        checkpoint_path = os.path.join(FLAGS.model_dir,
                                                       "parsing.ckpt")
                        model.saver.save(sess, checkpoint_path)
                        global_best_eval_ppx = best_eval_ppx

                        # record configuration
                        config_log_file = os.path.join(FLAGS.model_dir,
                                                       'config')
                        with open(config_log_file, 'w') as f:
                            d = FLAGS.__dict__['__flags']
                            for flag in sorted(d):
                                f.write("%s=%s\n" % (flag, str(d[flag])))

                for handler in logger.handlers:
                    handler.flush()
                sys.stdout.flush()
                step_time, loss, gradient_norm = 0.0, 0.0, 0.0

        if FLAGS.mode == 'train':
            logger.info('# steps to achieve minimum evaluation loss: %d' %
                        FLAGS.__dict__['__flags']['best_steps_on_validation'])
        logger.info('best evaluation perplexity of current configuration: '
                    '%.4f' % best_eval_ppx)
        logger.info('global best evaluation perplexity %.4f' %
                    global_best_eval_ppx)
        # reset FLAGS to default
        for key in default_flags:
            setattr(FLAGS, key, default_flags[key])
        return global_best_eval_ppx, best_eval_ppx


def test_rank():
    """Test a trained model as a ranker (as opposed to a generator)"""
    log_file = (FLAGS.test_log if (FLAGS.test_log is not None and
                                   FLAGS.test_log != 'None')
                else 'test.log')
    utils.setup_logging(log_dir=FLAGS.model_dir,
                        log_file=log_file)
    logger = logging.getLogger(__name__)

    # load vocabulary and test data
    vocab_path = os.path.join(FLAGS.data_dir, FLAGS.vocab_file)
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
    logger.info('loaded vocabulary from: %s. vocab size = %d' %
                (vocab_path, len(vocab)))
    if FLAGS.test_data_dir is not None and FLAGS.test_data_dir != 'None':
        # load testing data from another specified dir instead of the default
        logger.info('loading test data from: %s' % FLAGS.test_data_dir)
        source = []
        target = []
        source_test_path = os.path.join(FLAGS.test_data_dir, "source.test")
        target_test_path = os.path.join(FLAGS.test_data_dir, "target.test")
        max_source_length = 0
        max_target_length = 0
        with open(source_test_path, 'r') as f:
            for line in f:
                line = line.strip('\n')
                ids = data_utils.sentence_to_token_ids(line, vocab)
                source.append(ids)
                if len(ids) > max_source_length:
                    max_source_length = len(ids)
        with open(target_test_path, 'r') as f:
            for line in f:
                line = line.strip('\n')
                ids = data_utils.sentence_to_token_ids(line, vocab)
                ids.append(data_utils.EOS_ID)
                target.append(ids)
                if len(ids) > max_target_length:
                    max_target_length = len(ids)
        test_set = zip(source, target)
        FLAGS.encoder_size = max_source_length
        FLAGS.decoder_size = max_target_length + 1      # add 1 for _GO
    else:
        logger.info('loading test data from: %s' % FLAGS.data_dir)
        source_paths, target_paths, FLAGS.vocab_size = \
            data_utils.prepare_data(FLAGS.data_dir,
                                    FLAGS.vocab_file,
                                    FLAGS.vocab_size)
        _, _, source_test_path = source_paths
        _, _, target_test_path = target_paths
        # use the training set for testing to check sanity
        # source_test_path, _, _ = source_paths
        # target_test_path, _, _ = target_paths
        test_set, FLAGS.encoder_size, FLAGS.decoder_size = \
            read_data(source_test_path, target_test_path)
        FLAGS.decoder_size += 1     # add 1 for _GO

    # read ranking candidates, i.e., all the canonical utterances/logical forms
    if FLAGS.test_data_dir is not None and FLAGS.test_data_dir != 'None':
        candidate_path = os.path.join(FLAGS.test_data_dir, "candidates.json")
    else:
        candidate_path = os.path.join(FLAGS.data_dir, "candidates.json")
    with open(candidate_path, "r") as f:
        candidates = json.load(f)
    for cand in candidates:
        if FLAGS.eval_with_denotation:
            assert('denotation' in cand)
            cand['denotation'] = set(cand['denotation'])
        cand_ids = data_utils.sentence_to_token_ids(
            cand['canonical_utterance'], vocab) + [data_utils.EOS_ID]
        cand['ids'] = cand_ids
    n_cand = len(candidates)
    max_cand_length = max([len(cand['ids']) for cand in candidates])
    # TODO(ysu): remove these ugly encoder_size and decoder_size all together
    if FLAGS.decoder_size < max_cand_length + 1:
        FLAGS.decoder_size = max_cand_length + 1
    logger.info("# test cases: %d" % len(test_set))
    logger.info('# candidates: %d' % n_cand)
    logger.info("encoder size: %d. decoder size: %d." %
                (FLAGS.encoder_size, FLAGS.decoder_size))

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Create model and load parameters.
        model = create_model(sess)

        counter = 0
        mrr_avg = 0.0
        acc_avg = 0.0
        hit_avg = 0.0
        # parameter for metric HIT@K
        k = 3
        for utter, gold in test_set:
            found_gold = False
            for gold_id in xrange(n_cand):
                if candidates[gold_id]['ids'] == gold:
                    found_gold = True
                    break
            if not found_gold:
                logger.error("Gold canonical utterance not found: %s" %
                             gold)
                sys.exit(-1)

            data = [(utter, cand['ids']) for cand in candidates]
            (encoder_inputs, decoder_inputs, targets, target_weights,
             encoder_sequence_length, decoder_sequence_length) = \
                model.get_batch(data, batch_size=len(data))
            _, _, logits = model.step(sess,
                                      encoder_inputs,
                                      decoder_inputs,
                                      targets,
                                      target_weights,
                                      encoder_sequence_length,
                                      decoder_sequence_length,
                                      input_keep_prob=1.0,
                                      output_keep_prob=1.0,
                                      mode='test_rank')
            scores = [0] * n_cand
            for cand_id, cand in enumerate(candidates):
                ids = cand['ids']
                for step_id, cand_word in enumerate(ids):
                    step_logits = logits[cand_id][step_id]
                    step_probs = utils.softmax(step_logits)
                    scores[cand_id] += math.log(step_probs[cand_word])
            sorted_scores = [(i[0], i[1])
                             for i in sorted(enumerate(scores),
                                             key=lambda x:x[1],
                                             reverse=True)]

            def mrr(scores, gold_id):
                """Compute Mean Reciprocal Rank"""
                for pos, (cand_id, _) in enumerate(scores):
                    if cand_id == gold_id:
                        return 1.0 / (pos + 1)

            def accuracy(scores, gold_id):
                """Compute accuracy"""
                if scores[0][0] == gold_id:
                    return 1.0
                else:
                    return 0.0

            def hitatk(scores, gold_id, k):
                """Compute HIT@K. A hit is counted if the gold_id is in top k."""
                for i in range(k):
                    if scores[i][0] == gold_id:
                        return 1.0
                else:
                    return 0.0

            if FLAGS.eval_with_denotation:
                # if using denotation for evaluation, we use the top-ranked
                # candidate that has the same denotation as the gold
                # candidate (can make a difference in the overnight data)
                gold_denotation = candidates[gold_id]['denotation']
                # if len(gold_denotation) > 0:
                for cand_id, _ in sorted_scores:
                    cand_denotation = candidates[cand_id]['denotation']
                    if cand_denotation == gold_denotation:
                        gold_id = cand_id
                        break
                # else:
                #     logger.info('%s: gold denotation is %s, so we resort to '
                #                 'using canonical utterances for evaluation.' %
                #                 (candidates[gold_id]['canonical_utterance'],
                #                  gold_denotation))
            (gold_rank,) = [idx for idx, (cand_id, score)
                            in enumerate(sorted_scores)
                            if cand_id == gold_id]
            mrr_single = mrr(sorted_scores, gold_id)
            mrr_avg += mrr_single
            acc_single = accuracy(sorted_scores, gold_id)
            acc_avg += acc_single
            hit = hitatk(sorted_scores, gold_id, k)
            hit_avg += hit
            utter_text = [rev_vocab[id] for id in utter]
            gold_text = [rev_vocab[id] for id in gold]
            top_candidate = candidates[sorted_scores[0][0]]['ids']
            top_candidate_text = [rev_vocab[id] for id in top_candidate]
            logger.info("test case %d" % counter)
            logger.info("input utterance: %s" % utter_text)
            logger.info("gold canonical utterance: %s" % gold_text)
            logger.info("top canonical utterance: %s" % top_candidate_text)
            logger.info("gold rank = %d, top score = %.4f, "
                        "gold score = %.4f." %
                        (gold_rank, sorted_scores[0][1],
                         sorted_scores[gold_rank][1]))
            logger.info("mrr: %.4f. acc: %.1f. hit@%d: %.1f" %
                        (mrr_single, acc_single, k, hit))
            counter += 1
        mrr_avg /= counter
        acc_avg /= counter
        hit_avg /= counter
        mrr_random_guess = np.mean([1.0 / (i + 1) for i in xrange(n_cand)])
        acc_random_guess = 1.0 / n_cand
        logger.info("%d test cases, mean acc = %.4f; mean mrr = %.4f. "
                    " mean hit@%d: %.4f. %d candidates. "
                    " ramdom guessing acc = %.4f; mrr = %.4f." %
                    (counter, acc_avg, mrr_avg, k, hit_avg,
                     n_cand, acc_random_guess, mrr_random_guess))


def main(_):
    if FLAGS.mode == "train" or FLAGS.mode == "retrain":
        # Now only support direct call to train_grid()
        pass
    elif FLAGS.mode == "test_rank":
        test_rank()
    else:
        raise ValueError("Undefined mode")


if __name__ == "__main__":
    tf.app.run()
