from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip     # pylint: disable=redefined-builtin

from tensorflow.python import shape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
import tensorflow as tf

linear = rnn_cell._linear  # pylint: disable=protected-access


def _extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):
    """Get a loop_function that extracts the previous symbol and embeds it.

    Args:
        embedding: embedding tensor for symbols.
        output_projection: None or a pair (W, B). If provided, each fed
            previous output will first be multiplied by W and added B.
        update_embedding: Boolean; if False, the gradients will not propagate
            through the embeddings.

    Returns:
        A loop function.
    """
    def loop_function(prev, _):
        if output_projection is not None:
            prev = nn_ops.xw_plus_b(
                prev, output_projection[0], output_projection[1])
        prev_symbol = math_ops.argmax(prev, 1)
        # Note that gradients will not propagate through the second parameter
        # of embedding_lookup.
        emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = array_ops.stop_gradient(emb_prev)
        return emb_prev
    return loop_function


def rnn_decoder(cell, decoder_inputs, initial_state,
                sequence_length=None, loop_function=None,
                scope=None, dtype=None):
    """RNN decoder for the sequence-to-sequence model.

    Args:
        decoder_inputs: A list of 2D Tensors [batch_size x input_size].
        initial_state: 2D Tensor with shape [batch_size x cell.state_size].
        cell: rnn_cell.RNNCell defining the cell function and size.
        loop_function: If not None, this function will be applied to the i-th
        output in order to generate the i+1-st input, and decoder_inputs will
        be ignored, except for the first element ("GO" symbol). This can be
        used for decoding, but also for training to emulate
        http://arxiv.org/abs/1506.03099.
            Signature -- loop_function(prev, i) = next
                * prev is a 2D Tensor of shape [batch_size x output_size],
                * i is an integer, the step number (when advanced control is
                    needed),
                * next is a 2D Tensor of shape [batch_size x input_size].
        scope: VariableScope for the created subgraph; defaults to
        "rnn_decoder".

    Returns:
        A tuple of the form (outputs, state), where:
            outputs: A list of the same length as decoder_inputs of 2D Tensors
                with shape [batch_size x output_size] containing generated
                outputs.
            state: The state of each cell at the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
                (Note that in some cases, like basic RNN cell or GRU cell,
                outputs and states can be the same. They are different for
                LSTM cells though.)
    """
    # TODO(ysu): feed_previous using loop function
    outputs, state = rnn.dynamic_rnn(cell, decoder_inputs,
                                     sequence_length=sequence_length,
                                     initial_state=initial_state,
                                     scope=scope, dtype=dtype)
    # fetch the real outputs of the last unit and put into a single tensor
    # of shape [batch_size, num_decoder_symbols]. Not necessary for now.
    # outputs = tf.concat(0, outputs)
    # sequence_length = sequence_length - 1
    # batch_size = tf.shape(encoder_inputs[0])[0]
    # offset = sample_length * batch_size
    # indices = tf.range(tf.size(sequence_length)) + offset
    # outputs = tf.gather(outputs, indices)
    return outputs, state


def attention_decoder(decoder_inputs, initial_state, attention_states, cell,
                      output_size=None, num_heads=1, loop_function=None,
                      dtype=dtypes.float32, scope=None,
                      initial_state_attention=False):
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention "
                         "decoder.")
    if not attention_states.get_shape()[1:2].is_fully_defined():
        raise ValueError("Shape[1] and [2] of attention_states must be "
                         "known: %s" % attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with variable_scope.variable_scope(scope or "attention_decoder"):
        batch_size = array_ops.shape(decoder_inputs[0])[0]
        attn_length = attention_states.get_shape()[1].value
        attn_size = attention_states.get_shape()[2].value

        hidden = array_ops.reshape(
            attention_states, [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []
        attention_vec_size = attn_size  # Size of query vectors for attention.
        for a in xrange(num_heads):
            k = variable_scope.get_variable(
                "AttnW_%d" % a, [1, 1, attn_size, attention_vec_size])
            hidden_features.append(nn_ops.conv2d(hidden, k,
                                                 [1, 1, 1, 1], "SAME"))
            v.append(variable_scope.get_variable(
                "AttnV_%d" % a, [attention_vec_size]))
        state = initial_state

        def attention(query):
            ds = []  # Results of attention reads will be stored here.
            if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                query_list = nest.flatten(query)
                for q in query_list:  # Check that ndims == 2 if specified.
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = array_ops.concat(1, query_list)
            for a in xrange(num_heads):
                with variable_scope.variable_scope("Attention_%d" % a):
                    y = linear(query, attention_vec_size, True)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = math_ops.reduce_sum(
                        v[a] * math_ops.tanh(hidden_features[a] + y),
                        [2, 3])
                    a = nn_ops.softmax(s)
                    # Now calculate the attention-weighted vector d.
                    d = math_ops.reduce_sum(
                        array_ops.reshape(a, [-1, attn_length, 1, 1]) *
                        hidden, [1, 2])
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
            return ds

        outputs = []
        prev = None
        batch_attn_size = array_ops.pack([batch_size, attn_size])
        attns = [array_ops.zeros(batch_attn_size, dtype=dtype)
                 for _ in xrange(num_heads)]
        for a in attns:  # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])
        if initial_state_attention:
            attns = attention(initial_state)
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function",
                                                   reuse=True):
                    inp = loop_function(prev, i)
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" %
                                 inp.name)
            x = linear([inp] + attns, input_size, True)
            # Run the RNN.
            cell_output, state = cell(x, state)
            # Run the attention mechanism.
            if i == 0 and initial_state_attention:
                with variable_scope.variable_scope(
                        variable_scope.get_variable_scope(), reuse=True):
                    attns = attention(state)
            else:
                attns = attention(state)

            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + attns, output_size, True)
            if loop_function is not None:
                prev = output
            outputs.append(output)

    return outputs, state


def bidirectional_dynamic_attention_seq2seq(encoder_inputs,
                                            decoder_inputs,
                                            cell,
                                            num_decoder_symbols,
                                            encoder_sequence_length=None,
                                            decoder_sequence_length=None,
                                            feed_previous=False,
                                            scope=None,
                                            dtype=None):
    with variable_scope.variable_scope(
            scope or 'bidirectional_dynamic_attention_seq2seq') as scope:
        if dtype is not None:
            scope.set_dtype(dtype)
        else:
            dtype = scope.dtype

        # Encoder.
        encoder_states, encoder_final_state = \
            rnn.bidirectional_dynamic_rnn(cell, cell, encoder_inputs,
                sequence_length=encoder_sequence_length,
                scope='bidirectional_dynamic_rnn', dtype=dtype)
        encoder_states = tf.concat(2, encoder_states)
        encoder_final_state = tf.concat(1, encoder_final_state)
        with variable_scope.variable_scope('state_transform') as scope:
            W_t = tf.get_variable('W_t',
                                  [encoder_final_state.get_shape()[1],
                                   cell.state_size])
            b_t = tf.get_variable('b_t', [cell.state_size])
            decoder_initial_state = \
                math_ops.matmul(encoder_final_state, W_t) + b_t
            decoder_initial_state = math_ops.tanh(decoder_initial_state)

        return attention_decoder(decoder_inputs, decoder_initial_state,
                                 encoder_states, cell,
                                 scope='attention_decoder', dtype=dtype)


def dynamic_rnn_seq2seq(encoder_inputs,
                        decoder_inputs,
                        cell,
                        num_decoder_symbols,
                        encoder_sequence_length=None,
                        decoder_sequence_length=None,
                        feed_previous=False,
                        scope=None,
                        dtype=None):
    """Embedding RNN sequence-to-sequence model.

    This model first embeds encoder_inputs by a newly created embedding
    (of shape [num_encoder_symbols x input_size]). Then it runs an RNN to
    encode embedded encoder_inputs into a state vector. Next, it embeds
    decoder_inputs by another newly created embedding (of shape
    [num_decoder_symbols x input_size]). Then it runs RNN decoder, initialized
    with the last encoder state, on embedded decoder_inputs.

    Args:
        encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
        decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
        cell: rnn_cell.RNNCell defining the cell function and size.
        num_encoder_symbols: Integer; number of symbols on the encoder side.
        num_decoder_symbols: Integer; number of symbols on the decoder side.
        embedding_size: Integer, the length of the embedding vector for each
            symbol.
        output_projection: None or a pair (W, B) of output projection weights
            and biases; W has shape [output_size x num_decoder_symbols] and B
            has shape [num_decoder_symbols]; if provided and
            feed_previous=True, each fed previous output will first be
            multiplied by W and added B.
        feed_previous: Boolean or scalar Boolean Tensor; if True, only the
            first of decoder_inputs will be used (the "GO" symbol), and all
            other decoder inputs will be taken from previous outputs (as in
            embedding_rnn_decoder). If False, decoder_inputs are used as given
            (the standard decoder case).
        dtype: The dtype of the initial state for both the encoder and encoder
            rnn cells (default: tf.float32).
        scope: VariableScope for the created subgraph; defaults to
            "embedding_rnn_seq2seq"

    Returns:
        A tuple of the form (outputs, state), where:
            outputs: A list of the same length as decoder_inputs of 2D
                Tensors. The output is of shape [batch_size x
                cell.output_size] when output_projection is not None (and
                represents the dense representation of predicted tokens).
                It is of shape [batch_size x num_decoder_symbols]
                when output_projection is None.
            state: The state of each decoder cell in each time-step. This is a
                list with length len(decoder_inputs) -- one item for each
                time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
    """
    with variable_scope.variable_scope(scope or
                                       "dynamic_rnn_seq2seq") as scope:
        if dtype is not None:
            scope.set_dtype(dtype)
        else:
            dtype = scope.dtype

        # Encoder.
        _, encoder_state = \
            rnn.dynamic_rnn(cell, encoder_inputs,
                            sequence_length=encoder_sequence_length,
                            scope='dynamic_rnn_encoder', dtype=dtype)
        # import tensorflow as tf
        # tf.histogram_summary(encoder_state.name, encoder_state,
        #                      name='encoder_final_state')

        # TODO(ysu): embedding and feed_previous
        # loop_function = _extract_argmax_and_embed(
        #     embedding, output_projection,
        #     update_embedding_for_previous) if feed_previous else None
        return rnn_decoder(cell, decoder_inputs, encoder_state,
                           sequence_length=decoder_sequence_length,
                           scope='dynamic_rnn_decoder', dtype=dtype)


def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
    """Weighted cross-entropy loss for a sequence of logits (per example).

    Args:
        logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
        targets: List of 1D batch-sized int32 Tensors of the same length as
            logits.
        weights: List of 1D batch-sized float-Tensors of the same length as
            logits.
        average_across_timesteps: If set, divide the returned cost by the total
            label weight.
        softmax_loss_function: Function (inputs-batch, labels-batch) ->
            loss-batch to be used instead of the standard softmax (the default
            if this is None).
        name: Optional name for this operation, default:
            "sequence_loss_by_example".

    Returns:
        1D batch-sized float Tensor: The log-perplexity for each sequence.

    Raises:
        ValueError: If len(logits) is different from len(targets) or
            len(weights).
    """
    if len(targets) != len(logits) or len(weights) != len(logits):
        raise ValueError("Lengths of logits, weights, and targets must be the "
                         "same %d, %d, %d." %
                         (len(logits), len(weights), len(targets)))
    with ops.name_scope(name, "sequence_loss_by_example",
                        logits + targets + weights):
        log_perp_list = []
        for logit, target, weight in zip(logits, targets, weights):
            if softmax_loss_function is None:
                # TODO(irving,ebrevdo): This reshape is needed because
                # sequence_loss_by_example is called with scalars sometimes,
                # which violates our general scalar strictness policy.
                target = array_ops.reshape(target, [-1])
                crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
                    logit, target)
            else:
                crossent = softmax_loss_function(logit, target)
            log_perp_list.append(crossent * weight)
        log_perps = math_ops.add_n(log_perp_list)
        if average_across_timesteps:
            total_size = math_ops.add_n(weights)
            # Just to avoid division by 0 for all-0 weights.
            total_size += 1e-12
            log_perps /= total_size
    return log_perps


def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True,
                  average_across_batch=True,
                  softmax_loss_function=None, name=None):
    """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

    Args:
        logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
        targets: List of 1D batch-sized int32 Tensors of the same length as
            logits.
        weights: List of 1D batch-sized float-Tensors of the same length as
            logits.
        average_across_timesteps: If set, divide the returned cost by the total
            label weight.
        average_across_batch: If set, divide the returned cost by the batch
            size.
        softmax_loss_function: Function (inputs-batch, labels-batch) ->
            loss-batch to be used instead of the standard softmax (the default
            if this is None).
        name: Optional name for this operation, defaults to "sequence_loss".

    Returns:
        A scalar float Tensor: The average log-perplexity per symbol
            (weighted).

    Raises:
        ValueError: If len(logits) is different from len(targets) or
            len(weights).
    """
    with ops.name_scope(name, "sequence_loss", logits + targets + weights):
        cost = math_ops.reduce_sum(sequence_loss_by_example(
            logits, targets, weights,
            average_across_timesteps=average_across_timesteps,
            softmax_loss_function=softmax_loss_function))
        if average_across_batch:
            batch_size = array_ops.shape(targets[0])[0]
            return cost / math_ops.cast(batch_size, cost.dtype)
        else:
            return cost
