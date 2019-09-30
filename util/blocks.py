"""

Functions and components that can be slotted into tensorflow models.

TODO: Write functions for various types of attention.

"""

import tensorflow as tf


def length(sequence):
    """
    Get true length of sequences (without padding), and mask for true-length in max-length.

    Input of shape: (batch_size, max_seq_length)
    Output shapes, 
    length: (batch_size)
    mask: (batch_size, max_seq_length, 1)
    """
    populated = tf.sign(tf.abs(sequence))
    length = tf.cast(tf.reduce_sum(populated, axis=1), tf.int32)
    mask = tf.cast(tf.expand_dims(populated, -1), tf.float32)
    return length, mask



def biLSTM(inputs, dim, seq_len, name):
    """
    A Bi-Directional LSTM layer. Returns forward and backward hidden states as a tuple, and cell states as a tuple.

    Ouput of hidden states: [(batch_size, max_seq_length, hidden_dim), (batch_size, max_seq_length, hidden_dim)]
    Same shape for cell states.
    """
    with tf.name_scope(name):
        with tf.variable_scope('forward' + name):
            lstm_fwd = tf.contrib.rnn.LSTMCell(num_units=dim)
        with tf.variable_scope('backward' + name):
            lstm_bwd = tf.contrib.rnn.LSTMCell(num_units=dim)

        hidden_states, cell_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fwd, cell_bw=lstm_bwd, inputs=inputs, sequence_length=seq_len, dtype=tf.float32, scope=name)

    return hidden_states, cell_states


def LSTM(inputs, dim, seq_len, name):
    """
    An LSTM layer. Returns hidden states and cell states as a tuple.

    Ouput shape of hidden states: (batch_size, max_seq_length, hidden_dim)
    Same shape for cell states.
    """
    with tf.name_scope(name):
        cell = tf.contrib.rnn.LSTMCell(num_units=dim)
        hidden_states, cell_states = tf.nn.dynamic_rnn(cell, inputs=inputs, sequence_length=seq_len, dtype=tf.float32, scope=name)

    return hidden_states, cell_states

# Mask: (Bash, Length)
def attention(inputs, attention_size, time_major=False, return_alphas=False, mask=None):

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer



    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    if mask is not None:
        vu = vu * mask

    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas


def last_output(output, true_length):
    """
    To get the last hidden layer form a dynamically unrolled RNN.
    Input of shape (batch_size, max_seq_length, hidden_dim).

    true_length: Tensor of shape (batch_size). Such a tensor is given by the length() function.
    Output of shape (batch_size, hidden_dim).
    """
    max_length = int(output.get_shape()[1])
    length_mask = tf.expand_dims(tf.one_hot(true_length-1, max_length, on_value=1., off_value=0.), -1)
    last_output = tf.reduce_sum(tf.multiply(output, length_mask), 1)
    return last_output


def masked_softmax(scores, mask):
    """
    Used to calculcate a softmax score with true sequence length (without padding), rather than max-sequence length.

    Input shape: (batch_size, max_seq_length, hidden_dim). 
    mask parameter: Tensor of shape (batch_size, max_seq_length). Such a mask is given by the length() function.
    """
    numerator = tf.exp(tf.subtract(scores, tf.reduce_max(scores, 1, keep_dims=True))) * mask
    denominator = tf.reduce_sum(numerator, 1, keep_dims=True)
    weights = tf.div(numerator, denominator)
    return weights
