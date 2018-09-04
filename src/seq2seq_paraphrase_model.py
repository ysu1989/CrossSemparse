"""Sequence-to-sequence paraphrasing model."""

import random
import logging
import numpy as np
import math
import tensorflow as tf

import data_utils
import seq2seq


class Seq2SeqParaphraseModel(object):
    """Sequence-to-sequence paraphrasing model."""

    def __init__(self,
                 encoder_size,
                 decoder_size,
                 vocab_size,
                 size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 optimization_algorithm,
                 learning_rate,
                 learning_rate_decay_factor,
                 use_lstm=False,
                 use_attention=False,
                 use_word2vec=False,
                 embedding_size=-1,
                 initial_embeddings=None,
                 train_word2vec_embedding=True,
                 summarize_trainable_variables=False,
                 adam_epsilon=1e-8):
        """Create the model."""
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.optimization_algorithm = optimization_algorithm
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.logger = logging.getLogger(__name__)

        # Create the internal multi-layer cell for our RNN.
        single_cell = tf.nn.rnn_cell.GRUCell(size)
        if use_lstm:
            single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
        self.input_keep_prob = tf.placeholder(tf.float32,
                                              shape=[],
                                              name='input_keep_prob')
        self.output_keep_prob = tf.placeholder(tf.float32,
                                               shape=[],
                                               name='output_keep_prob')
        single_cell = tf.nn.rnn_cell.DropoutWrapper(
            single_cell, input_keep_prob=self.input_keep_prob,
            output_keep_prob=self.output_keep_prob)
        cell = single_cell
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

        # The seq2seq function.
        def seq2seq_f(encoder_inputs, decoder_inputs):
            if use_attention:
                self.logger.info(
                    "creating bidirectional_dynamic_attention_seq2seq")
                # need to unpack decoder inputs because the current
                # attention decoder uses static rnn instead of dynamic rnn
                decoder_inputs = tf.unpack(decoder_inputs, axis=1)
                return seq2seq.bidirectional_dynamic_attention_seq2seq(
                    encoder_inputs,
                    decoder_inputs,
                    cell,
                    vocab_size,
                    encoder_sequence_length=self.encoder_sequence_length,
                    decoder_sequence_length=self.decoder_sequence_length)
            else:
                self.logger.info("creating dynamic_rnn_seq2seq")
                return seq2seq.dynamic_rnn_seq2seq(
                    encoder_inputs,
                    decoder_inputs,
                    cell,
                    vocab_size,
                    encoder_sequence_length=self.encoder_sequence_length,
                    decoder_sequence_length=self.decoder_sequence_length)

        # word embedding
        # TODO(ysu): set the embedding of _PAD as all-zero, otherwise the
        # padding will make undesired impact if we don't do dynamic unrolling
        with tf.device('/gpu:0'):
            if use_word2vec and initial_embeddings is not None:
                initializer = tf.constant_initializer(initial_embeddings)
            else:
                # Initializer for embeddings should have variance=1.
                sqrt3 = math.sqrt(3)
                initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)
            if use_word2vec:
                train_embedding = train_word2vec_embedding
            else:
                # always train embeddings when not using word2vec
                train_embedding = True
            embedding_size = size if embedding_size == -1 else embedding_size
            self.embedding = tf.get_variable("embedding",
                                             [self.vocab_size, embedding_size],
                                             initializer=initializer,
                                             trainable=train_embedding)
            # set the embedding of PAD to all zero
            self.embedding = tf.scatter_update(self.embedding,
                                               tf.constant(data_utils.PAD_ID),
                                               tf.constant([0.0] *
                                                           embedding_size))

        # Feeds for inputs.
        self.encoder_inputs = tf.placeholder(tf.int32,
                                             shape=[None, self.encoder_size],
                                             name='encoder_inputs')
        with tf.device('/gpu:0'):
            self.embedded_encoder_inputs = \
                tf.nn.embedding_lookup(self.embedding,
                                       self.encoder_inputs)
        self.decoder_inputs = tf.placeholder(tf.int32,
                                             shape=[None, self.decoder_size],
                                             name='decoder_inputs')
        with tf.device('/gpu:0'):
            self.embedded_decoder_inputs = \
                tf.nn.embedding_lookup(self.embedding,
                                       self.decoder_inputs)
        self.targets = tf.placeholder(tf.int32,
                                      shape=[None, self.decoder_size],
                                      name='decoder_inputs')
        self.target_weights = tf.placeholder(tf.float32,
                                             shape=[None, self.decoder_size],
                                             name='decoder_inputs')
        self.encoder_sequence_length = \
            tf.placeholder(tf.int32, shape=[None],
                           name='encoder_sequence_length')
        self.decoder_sequence_length = \
            tf.placeholder(tf.int32, shape=[None],
                           name='decoder_sequence_length')

        def _linear(inputs, output_size, dtype=tf.float32, scope=None):
            input_size = inputs.get_shape().as_list()[1]
            with tf.variable_scope(scope or "Linear"):
                weight = tf.get_variable(
                    "weight", [input_size, output_size], dtype=dtype)
                bias = tf.get_variable(
                    'bias',
                    [output_size],
                    initializer=tf.constant_initializer(
                        0.0, dtype=dtype))
                output = tf.matmul(inputs, weight) + bias
            return output

        def _feedforward_nn(inputs, output_size, dtype=tf.float32, scope=None):
            with tf.variable_scope(scope or "FFNN"):
                with tf.variable_scope('L1') as scope:
                    z1 = _linear(inputs, output_size, dtype=dtype, scope=scope)
                    # tf.histogram_summary(z1.name, z1)
                    a1 = tf.tanh(z1)
                    # a1 = tf.nn.dropout(a1, self.output_keep_prob)
                with tf.variable_scope('L2') as scope:
                    z2 = _linear(a1, output_size, dtype=dtype, scope=scope)
                    # tf.histogram_summary(z2.name, z2)
                # with tf.variable_scope('L2') as scope:
                #     z2 = _linear(a1, output_size, dtype=dtype, scope=scope)
                #     tf.histogram_summary(z2.name, z2)
                #     a2 = tf.tanh(z2)
                #     a2 = tf.nn.dropout(a2, self.output_keep_prob)
                #     tf.histogram_summary(a2.name, a2)
                # with tf.variable_scope('L3') as scope:
                #     z3 = _linear(a2, output_size, dtype=dtype, scope=scope)
                return z2

        # Training outputs
        self.decoder_states, _ = seq2seq_f(self.embedded_encoder_inputs,
                                           self.embedded_decoder_inputs)
        # tf.histogram_summary(self.decoder_states.name, self.decoder_states,
        #                      name='decoder_final_state')

        # dynamic rnn decoder
        # TODO(ysu): write a sequence loss function for outputs of dynamic
        #   rnn. currently we use unpacking, which requires we explicitly
        #   specify the maximum sequence length during graph construction,
        #   so we are not really unrolling the rnn to the maximum sequence
        #   length of some batch of data, but to the global maxmimum
        #   sequence length.
        decoder_states = self.decoder_states
        if not use_attention:
            decoder_states = tf.unpack(self.decoder_states, axis=1)

        # output_projection = _feedforward_nn
        output_projection = _linear
        with tf.variable_scope('output_projection') as scope:
            outputs = []
            for i, state in enumerate(decoder_states):
                if i > 0:
                    scope.reuse_variables()
                outputs.append(output_projection(
                    decoder_states[i], self.vocab_size,
                    scope=scope))
            self.outputs = tf.pack(outputs, axis=1)

        targets = tf.unpack(self.targets, axis=1)
        target_weights = tf.unpack(self.target_weights, axis=1)
        self.loss = seq2seq.sequence_loss(outputs,
                                          targets,
                                          target_weights)

        # Gradients and SGD update operation for training the model.
        if self.optimization_algorithm == 'vanilla':
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.optimization_algorithm == 'adagrad':
            opt = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optimization_algorithm == 'rmsprop':
            opt = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optimization_algorithm == 'adadelta':
            # use default learning rate
            opt = tf.train.AdadeltaOptimizer(self.learning_rate)
        elif self.optimization_algorithm == 'adam':
            # use default learning rate
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                         epsilon=adam_epsilon)

        # 1) no gradient clipping
        # params = tf.trainable_variables()
        # gradients = tf.gradients(self.loss, params)
        # for gradient in gradients:
        #     tf.histogram_summary(gradient.name, gradient)
        # self.gradient_norm = tf.global_norm(gradients)
        # self.update = opt.minimize(self.loss,
        #                            global_step=self.global_step,
        #                            var_list=params)

        # 2) gradient clipping
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        # for gradient in gradients:
        #     tf.histogram_summary(gradient.name, gradient)
        clipped_gradients, norm = \
            tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.gradient_norm = norm
        self.update = opt.apply_gradients(zip(clipped_gradients, params),
                                          global_step=self.global_step)

        # summarize trainable variables
        if summarize_trainable_variables:
            variables = tf.trainable_variables()
            for v in variables:
                tf.histogram_summary(v.name, v)

        # don't save Adam parameters (not used in testing)
        # to save space and time
        vars_to_save = [var for var in tf.all_variables()
                        if 'Adam' not in var.name]
        self.logger.info('variables to save:')
        for var in vars_to_save:
            self.logger.info(var.name)
        self.saver = tf.train.Saver(vars_to_save, max_to_keep=1)
        # self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
        self.merged_summary = tf.merge_all_summaries()
        # we don't have the computation graph here
        self.summary_writer = None

    def step(self,
             session,
             encoder_inputs,
             decoder_inputs,
             targets,
             target_weights,
             encoder_sequence_length,
             decoder_sequence_length,
             input_keep_prob=1.0,
             output_keep_prob=1.0,
             mode='train'):
        """Run a step of the model feeding the given inputs."""
        input_feed = {}
        input_feed[self.encoder_inputs] = encoder_inputs
        input_feed[self.decoder_inputs] = decoder_inputs
        input_feed[self.targets] = targets
        input_feed[self.target_weights] = target_weights
        input_feed[self.encoder_sequence_length] = encoder_sequence_length
        input_feed[self.decoder_sequence_length] = decoder_sequence_length
        input_feed[self.input_keep_prob] = input_keep_prob
        input_feed[self.output_keep_prob] = output_keep_prob

        if mode == 'train':
            # training
            output_feed = [self.update,         # SGD
                           self.gradient_norm,  # Gradient norm
                           self.loss]           # Loss for this batch
            outputs = session.run(output_feed, input_feed)
            # Gradient norm, loss, no outputs
            return outputs[1], outputs[2], None
        elif mode == 'test_rank':
            # testing as a ranker
            # TODO(ysu): separate mode for validation
            output_feed = [self.loss,           # Loss for this batch
                           self.outputs]        # Output logits
            outputs = session.run(output_feed, input_feed)
            # No gradient norm, loss, outputs
            return None, outputs[0], outputs[1]
        elif mode == 'summarize':
            output_feed = self.merged_summary
            outputs = session.run(output_feed, input_feed)
            return outputs

    def get_batch(self, data, batch_size=None):
        """Get a random batch of data from the specified bucket for step."""
        if batch_size is None:
            # randomly generate a batch for training
            batch_size = self.batch_size
            random_sample = True
        else:
            # convert the whole 'data' into a batch
            # useful in validation or testing
            random_sample = False
        encoder_size, decoder_size = self.encoder_size, self.decoder_size
        # encoder_size = max([len(encoder_input) for encoder_input, _ in data])
        # decoder_size = max([len(decoder_input) for _, decoder_input in data])
        (batch_encoder_inputs, batch_decoder_inputs,
         encoder_sequence_length, decoder_sequence_length) = [], [], [], []

        for sample_id in xrange(batch_size):
            if random_sample:
                encoder_input, decoder_input = random.choice(data)
            else:
                encoder_input, decoder_input = data[sample_id]
            encoder_sequence_length.append(len(encoder_input))
            # add 1 for _Go
            decoder_sequence_length.append(len(decoder_input) + 1)

            # Encoder inputs are padded.
            encoder_pad = ([data_utils.PAD_ID] *
                           (encoder_size - len(encoder_input)))
            batch_encoder_inputs.append(encoder_input + encoder_pad)

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            batch_decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                                        [data_utils.PAD_ID] * decoder_pad_size)

        # Here the assumption is that data_utils._PAD = 0
        batch_targets = np.zeros([batch_size, decoder_size], dtype=np.int32)
        batch_weights = np.zeros([batch_size, decoder_size], dtype=np.float32)
        for length_idx in xrange(decoder_size):
            # Create target_weights to be 0 for targets that are padding.
            for batch_idx in xrange(batch_size):
                # We set weight to 0 if the corresponding target is a
                # PAD symbol.
                # The corresponding target is decoder_input shifted by
                # 1 forward.
                if length_idx < decoder_size - 1:
                    batch_targets[batch_idx][length_idx] = \
                        batch_decoder_inputs[batch_idx][length_idx + 1]
                if (length_idx < decoder_size - 1 and
                        batch_targets[batch_idx, length_idx] != data_utils.PAD_ID):
                    batch_weights[batch_idx][length_idx] = 1.0
        return (batch_encoder_inputs, batch_decoder_inputs,
                batch_targets, batch_weights,
                encoder_sequence_length, decoder_sequence_length)
