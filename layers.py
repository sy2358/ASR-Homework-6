import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell


def get_dropout_mask(keep_prob, shape):
  keep_prob = tf.convert_to_tensor(keep_prob)
  random_tensor = keep_prob + tf.random_uniform(shape)
  binary_tensor = tf.floor(random_tensor)
  dropout_mask = tf.inv(keep_prob) * binary_tensor
  return dropout_mask


class VariationalDropoutWrapper(RNNCell):
  def __init__(self, cell, batch_size, keep_prob):
    self._cell = cell
    self._output_dropout_mask = get_dropout_mask(keep_prob, [batch_size, cell.output_size])
    self._state_dropout_mask = get_dropout_mask(keep_prob, [batch_size, int(cell.state_size / 2)])

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    # TODO: suppport non-LSTM cells and state_is_tuple=True
    c, h = tf.split(1, 2, state)
    h *= self._state_dropout_mask
    state = tf.concat(1, [c, h])
    output, new_state = self._cell(inputs, state, scope)
    output *= self._output_dropout_mask
    return output, new_state