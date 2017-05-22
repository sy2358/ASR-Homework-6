import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple


def get_dropout_mask(keep_prob, shape):
  keep_prob = tf.convert_to_tensor(keep_prob)
  random_tensor = keep_prob + tf.random_uniform(shape)
  binary_tensor = tf.floor(random_tensor)
  dropout_mask = tf.reciprocal(keep_prob) * binary_tensor
  return dropout_mask


class VariationalDropoutWrapper(RNNCell):
  def __init__(self, cell, batch_size, keep_prob):
    self._cell = cell
    self._output_dropout_mask = get_dropout_mask(keep_prob, [batch_size, cell.output_size])
    self._state_dropout_mask = get_dropout_mask(keep_prob, [batch_size, cell.state_size[0]])

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    # TODO: suppport non-LSTM cells and state_is_tuple=True
    # states = (c, h)

    c, h = state[0], state[1]
    c *= self._state_dropout_mask
    # l = tf.unstack(states_, axis=0)
    # rnn_tuple_state = tuple(
    #     [tf.contrib.rnn.LSTMStateTuple(l[idx][0], l[idx][1])
    #      for idx in range(2)]
    # )
    state = tuple(LSTMStateTuple(c, h))
    output, new_state = self._cell(inputs, state, scope)
    output *= self._output_dropout_mask
    return output, new_state


# This is a minimal gist of what you'd have to
# add to TensorFlow code to implement zoneout.

# To see this in action, see zoneout_seq2seq.py

z_prob_cells = 0.05
z_prob_states = 0


# Wrapper for the TF RNN cell
# For an LSTM, the 'cell' is a tuple containing state and cell
# We use TF's dropout to implement zoneout
# https://github.com/teganmaharaj/zoneout/blob/master/zoneout_tensorflow.py
class ZoneoutWrapper(RNNCell):
    """Operator adding zoneout to all states (states+cells) of the given cell."""

    def __init__(self, cell, zoneout_prob, is_training=True, seed=None):
        if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
            raise TypeError("The parameter cell is not an RNNCell.")
        if (isinstance(zoneout_prob, float) and
                not (zoneout_prob >= 0.0 and zoneout_prob <= 1.0)):
            raise ValueError("Parameter zoneout_prob must be between 0 and 1: %d"
                             % zoneout_prob)
        self._cell = cell
        self._zoneout_prob = zoneout_prob
        self._seed = seed
        self.is_training = is_training

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        if isinstance(self.state_size, tuple) != isinstance(self._zoneout_prob, tuple):
            raise TypeError("Subdivided states need subdivided zoneouts.")
        if isinstance(self.state_size, tuple) and len(tuple(self.state_size)) != len(tuple(self._zoneout_prob)):
            raise ValueError("State and zoneout need equally many parts.")
        output, new_state = self._cell(inputs, state, scope)
        if isinstance(self.state_size, tuple):
            if self.is_training:
                new_state = tuple((1 - state_part_zoneout_prob) * tf.python.nn_ops.dropout(
                    new_state_part - state_part, (1 - state_part_zoneout_prob), seed=self._seed) + state_part
                                  for new_state_part, state_part, state_part_zoneout_prob in
                                  zip(new_state, state, self._zoneout_prob))
            else:
                new_state = tuple(state_part_zoneout_prob * state_part + (1 - state_part_zoneout_prob) * new_state_part
                                  for new_state_part, state_part, state_part_zoneout_prob in
                                  zip(new_state, state, self._zoneout_prob))
        else:
            state_part_zoneout_prob = self._zoneout_prob
            if self.is_training:
                new_state = (1 - state_part_zoneout_prob) * tf.python.nn_ops.dropout(
                    new_state - state, (1 - state_part_zoneout_prob), seed=self._seed) + state
            else:
                new_state = state_part_zoneout_prob * state + (1 - state_part_zoneout_prob) * new_state
        return output, new_state


# Wrap your cells like this
# cell = ZoneoutWrapper(tf.nn.rnn_cell.LSTMCell(hidden_units, initializer=random_uniform(), state_is_tuple=True),
#                       zoneout_prob=(z_prob_cells, z_prob_states))